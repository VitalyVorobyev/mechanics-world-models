"""Evaluate RSSM reconstruction and open-loop prediction quality."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rich.console import Console
from rich.table import Table
import torch
from torch.utils.data import DataLoader, Subset

from data import EpisodeSequenceDataset, ImageCropConfig
from data.image_transforms import add_crop_args, crop_config_from_args, crop_config_from_mapping
from models import VisualWorldModel, compute_latent_consistency_loss, foreground_reconstruction_mask
from train.train_rssm import resolve_device, set_seed
from viz.rssm_eval_viz import (
    evenly_spaced_timesteps,
    write_prediction_contact_sheet,
    write_prediction_video,
)


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for offline RSSM evaluation."""

    checkpoint_path: Path
    dataset_dir: Path
    output_dir: Path
    sequence_length: int | None = None
    batch_size: int = 16
    num_sequences: int | None = 64
    warmup_length: int = 5
    horizons: tuple[int, ...] = (1, 5, 10)
    seed: int = 0
    val_fraction: float | None = None
    split: str = "val"
    device: str = "auto"
    num_workers: int = 0
    num_visualizations: int = 3
    video_fps: int = 6
    sheet_timesteps: int = 8
    crop_config: ImageCropConfig | None = None


@dataclass(frozen=True)
class EvalRollout:
    """Deterministic eval tensors, all image tensors shaped ``[B, T, 3, H, W]``."""

    reconstructions: torch.Tensor
    one_step_predictions: torch.Tensor
    open_loop_predictions: torch.Tensor


@dataclass
class MetricAccumulator:
    """Accumulate pixel-wise squared error and element counts."""

    squared_error: float = 0.0
    count: int = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> None:
        """Add a prediction/target tensor pair to the accumulator."""

        if prediction.shape != target.shape:
            raise ValueError(f"shape mismatch: {prediction.shape} != {target.shape}")
        squared_error = (prediction - target) ** 2
        if weight is not None:
            squared_error = squared_error * weight
        self.squared_error += float(torch.sum(squared_error).detach().cpu())
        self.count += int(target.numel())

    def mean(self) -> float:
        """Return the accumulated MSE."""

        if self.count == 0:
            raise ValueError("cannot average an empty metric accumulator")
        return self.squared_error / self.count


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command line parser for RSSM evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--warmup-length", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-visualizations", type=int, default=3)
    parser.add_argument("--video-fps", type=int, default=6)
    parser.add_argument("--sheet-timesteps", type=int, default=8)
    add_crop_args(parser, default_mode=None)
    return parser


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    """Convert parsed CLI args to a typed config dataclass."""

    return EvalConfig(
        checkpoint_path=args.checkpoint_path,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_sequences=args.num_sequences,
        warmup_length=args.warmup_length,
        horizons=tuple(args.horizons),
        seed=args.seed,
        val_fraction=args.val_fraction,
        split=args.split,
        device=args.device,
        num_workers=args.num_workers,
        num_visualizations=args.num_visualizations,
        video_fps=args.video_fps,
        sheet_timesteps=args.sheet_timesteps,
        crop_config=crop_config_from_args(args),
    )


def evaluate_checkpoint(config: EvalConfig, console: Console | None = None) -> dict[str, float]:
    """Evaluate one checkpoint and write metrics plus optional visual artifacts."""

    console = console or Console()
    set_seed(config.seed)
    device = resolve_device(config.device)
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    checkpoint_config = dict(checkpoint.get("config", {}))
    sequence_length = resolve_sequence_length(config, checkpoint_config)
    val_fraction = resolve_val_fraction(config, checkpoint_config)
    crop_config = resolve_crop_config(config, checkpoint_config)
    validate_eval_config(config, sequence_length)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = EpisodeSequenceDataset(
        dataset_dir=config.dataset_dir,
        sequence_length=sequence_length,
        split=config.split,  # type: ignore[arg-type]
        val_fraction=val_fraction,
        seed=config.seed,
        crop_config=crop_config,
    )
    selected_indices = select_sequence_indices(
        total_sequences=len(dataset),
        num_sequences=config.num_sequences,
        seed=config.seed,
    )
    eval_dataset = Subset(dataset, selected_indices)
    loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    first_sample = dataset[selected_indices[0]]
    action_dim = int(first_sample["actions"].shape[-1])
    image_size = int(first_sample["observations"].shape[-1])
    model = load_model_from_checkpoint(
        checkpoint=checkpoint,
        checkpoint_config=checkpoint_config,
        action_dim=action_dim,
        image_size=image_size,
        device=device,
    )
    if not getattr(model, "latent_predictor_loaded", True):
        console.print(
            "[yellow]checkpoint does not contain latent_predictor parameters; "
            "latent_consistency_loss will be skipped.[/yellow]",
        )

    console.print(
        f"evaluating checkpoint={config.checkpoint_path} split={config.split} "
        f"sequences={len(eval_dataset)} warmup={config.warmup_length} "
        f"horizons={config.horizons} crop={crop_config}",
    )

    metrics = evaluate_loader(
        model=model,
        loader=loader,
        device=device,
        warmup_length=config.warmup_length,
        horizons=config.horizons,
        output_dir=config.output_dir,
        num_visualizations=config.num_visualizations,
        video_fps=config.video_fps,
        sheet_timesteps=config.sheet_timesteps,
    )
    metrics.update(
        {
            "num_sequences": float(len(eval_dataset)),
            "sequence_length": float(sequence_length),
            "warmup_length": float(config.warmup_length),
        },
    )
    write_metrics(config.output_dir / "metrics.json", metrics, config, selected_indices, crop_config)
    print_metrics_table(console, metrics)
    return metrics


@torch.no_grad()
def evaluate_loader(
    model: VisualWorldModel,
    loader: DataLoader,
    device: torch.device,
    warmup_length: int,
    horizons: Sequence[int],
    output_dir: Path,
    num_visualizations: int,
    video_fps: int,
    sheet_timesteps: int,
) -> dict[str, float]:
    """Evaluate batches and optionally write sequence-level diagnostics."""

    model.eval()
    metrics: dict[str, MetricAccumulator] = {
        "reconstruction_mse": MetricAccumulator(),
        "one_step_prediction_mse": MetricAccumulator(),
        "foreground_one_step_prediction_mse": MetricAccumulator(),
        "open_loop_future_mse": MetricAccumulator(),
        "foreground_open_loop_future_mse": MetricAccumulator(),
    }
    for horizon in horizons:
        metrics[f"open_loop_mse_h{horizon}"] = MetricAccumulator()
        metrics[f"foreground_open_loop_mse_h{horizon}"] = MetricAccumulator()
    latent_consistency_loss_total = 0.0
    latent_consistency_batches = 0
    compute_latent_metric = bool(getattr(model, "latent_predictor_loaded", True))

    saved_visualizations = 0
    for batch in loader:
        observations = batch["observations"].to(device)
        next_observations = batch["next_observations"].to(device)
        actions = batch["actions"].to(device)
        # observations/next_observations: [B, T, 3, H, W], actions: [B, T, A].
        # next_observations[:, t] is the target for obs_t --action_t--> obs_{t+1}.
        rollout = deterministic_eval_rollout(model, observations, actions, warmup_length)
        prediction_mask = foreground_reconstruction_mask(next_observations).detach()
        if compute_latent_metric:
            latent_outputs = model(
                observations=observations,
                actions=actions,
                next_observations=next_observations,
            )
            latent_consistency_loss = compute_latent_consistency_loss(
                latent_outputs,
                latent_consistency_weight=1.0,
            )
            latent_consistency_loss_total += float(latent_consistency_loss.detach().cpu())
            latent_consistency_batches += 1

        metrics["reconstruction_mse"].update(rollout.reconstructions, observations)
        metrics["one_step_prediction_mse"].update(rollout.one_step_predictions, next_observations)
        metrics["foreground_one_step_prediction_mse"].update(
            rollout.one_step_predictions,
            next_observations,
            prediction_mask,
        )
        open_loop_start = warmup_length - 1
        metrics["open_loop_future_mse"].update(
            rollout.open_loop_predictions[:, open_loop_start:],
            next_observations[:, open_loop_start:],
        )
        metrics["foreground_open_loop_future_mse"].update(
            rollout.open_loop_predictions[:, open_loop_start:],
            next_observations[:, open_loop_start:],
            prediction_mask[:, open_loop_start:],
        )
        for horizon in horizons:
            timestep = warmup_length + horizon - 2
            metrics[f"open_loop_mse_h{horizon}"].update(
                rollout.open_loop_predictions[:, timestep],
                next_observations[:, timestep],
            )
            metrics[f"foreground_open_loop_mse_h{horizon}"].update(
                rollout.open_loop_predictions[:, timestep],
                next_observations[:, timestep],
                prediction_mask[:, timestep],
            )

        if saved_visualizations < num_visualizations:
            saved_visualizations = write_batch_visualizations(
                output_dir=output_dir,
                observations=observations,
                next_observations=next_observations,
                rollout=rollout,
                start_index=saved_visualizations,
                max_count=num_visualizations,
                video_fps=video_fps,
                sheet_timesteps=sheet_timesteps,
                warmup_length=warmup_length,
            )

    results = {name: accumulator.mean() for name, accumulator in metrics.items()}
    if latent_consistency_batches > 0:
        results["latent_consistency_loss"] = latent_consistency_loss_total / latent_consistency_batches
    return results


@torch.no_grad()
def deterministic_eval_rollout(
    model: VisualWorldModel,
    observations: torch.Tensor,
    actions: torch.Tensor,
    warmup_length: int,
) -> EvalRollout:
    """Run deterministic posterior, 1-step prior, and open-loop RSSM rollouts.

    Args:
        observations: Image batch shaped ``[B, T, 3, H, W]``.
        actions: Action batch shaped ``[B, T, A]``.
        warmup_length: Number of initial steps that may use true observations.

    Returns:
        Reconstructions aligned to ``obs_t`` plus 1-step and open-loop
        predictions aligned to ``obs_{t+1}``, all shaped ``[B, T, 3, H, W]``.
    """

    embeddings = model.encoder(observations)
    posterior_features, next_prior_features = posterior_and_prior_features(model, embeddings, actions)
    open_loop_features = open_loop_features_from_context(model, embeddings, actions, warmup_length)
    return EvalRollout(
        reconstructions=model.decoder(posterior_features),
        one_step_predictions=model.decoder(next_prior_features),
        open_loop_predictions=model.decoder(open_loop_features),
    )


@torch.no_grad()
def deterministic_reconstruction(
    model: VisualWorldModel,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Decode posterior-mean reconstructions aligned to the same target timestep.

    Args:
        observations: Target image batch shaped ``[B, T, 3, H, W]`` in ``[0, 1]``.
        actions: Action batch shaped ``[B, T, A]``.

    Returns:
        Reconstructed observations shaped ``[B, T, 3, H, W]`` in ``[0, 1]``.
    """

    # This is posterior-only reconstruction: target obs_t and decoded recon_t
    # share the same timestep index. No future observations or open-loop prior
    # features are used here.
    embeddings = model.encoder(observations)
    posterior_features, _ = posterior_and_prior_features(model, embeddings, actions)
    return model.decoder(posterior_features)


def posterior_and_prior_features(
    model: VisualWorldModel,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic posterior features and one-step prior features.

    The posterior feature for index ``t`` reconstructs ``obs_t``. The returned
    prior feature for index ``t`` predicts ``obs_{t+1}`` from the posterior state
    at ``obs_t`` and ``action_t``.
    """

    if embeddings.ndim != 3 or actions.ndim != 3:
        raise ValueError("embeddings and actions must have shapes [B, T, E] and [B, T, A]")
    if embeddings.shape[:2] != actions.shape[:2]:
        raise ValueError("embeddings and actions must share [B, T]")

    batch_size, sequence_length = embeddings.shape[:2]
    rssm = model.rssm
    device = embeddings.device
    dtype = embeddings.dtype
    h_t = torch.zeros(batch_size, rssm.hidden_size, device=device, dtype=dtype)

    posterior_features: list[torch.Tensor] = []
    next_prior_features = model.transition_aligned_next_prior_features(embeddings, actions)
    for step in range(sequence_length):
        # h_t is the deterministic state for obs_t; it has not seen action_t.
        # posterior_mean is [B, Z] and posterior feature is [B, H + Z].
        posterior_input = torch.cat([h_t, embeddings[:, step]], dim=-1)
        posterior_mean, _ = rssm._dist_params(rssm.posterior(posterior_input))
        posterior_features.append(torch.cat([h_t, posterior_mean], dim=-1))

        # action_t advances the deterministic state to obs_{t+1}.
        recurrent_input = torch.cat([posterior_mean, actions[:, step]], dim=-1)
        h_t = rssm.recurrent(recurrent_input, h_t)

    return torch.stack(posterior_features, dim=1), next_prior_features


def open_loop_features_from_context(
    model: VisualWorldModel,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    warmup_length: int,
) -> torch.Tensor:
    """Roll out features using posterior context then prior-only prediction.

    Output index ``t`` predicts ``obs_{t+1}``. For steps
    ``t < warmup_length - 1``, the prediction is a context/teacher-forced
    transition. At ``t == warmup_length - 1``, the model predicts the first frame
    after the warmup window using ``action_t`` and then continues prior-only.
    """

    if warmup_length < 1:
        raise ValueError("warmup_length must be >= 1")
    if embeddings.ndim != 3 or actions.ndim != 3:
        raise ValueError("embeddings and actions must have shapes [B, T, E] and [B, T, A]")
    if embeddings.shape[:2] != actions.shape[:2]:
        raise ValueError("embeddings and actions must share [B, T]")
    if warmup_length >= embeddings.shape[1]:
        raise ValueError("warmup_length must be smaller than sequence length")

    batch_size, sequence_length = embeddings.shape[:2]
    rssm = model.rssm
    device = embeddings.device
    dtype = embeddings.dtype
    h_t = torch.zeros(batch_size, rssm.hidden_size, device=device, dtype=dtype)
    z_t = torch.zeros(batch_size, rssm.latent_size, device=device, dtype=dtype)
    features: list[torch.Tensor] = []

    for step in range(sequence_length):
        # h_t is the deterministic state for obs_t. During context, infer z_t
        # from embed_t; after context, use the prior z_t carried from the
        # previous transition.
        if step < warmup_length:
            posterior_input = torch.cat([h_t, embeddings[:, step]], dim=-1)
            posterior_mean, _ = rssm._dist_params(rssm.posterior(posterior_input))
            z_t = posterior_mean

        recurrent_input = torch.cat([z_t, actions[:, step]], dim=-1)
        h_t = rssm.recurrent(recurrent_input, h_t)
        prior_next_mean, _ = rssm._dist_params(rssm.prior(h_t))
        z_t = prior_next_mean
        features.append(torch.cat([h_t, z_t], dim=-1))

    return torch.stack(features, dim=1)


def write_batch_visualizations(
    output_dir: Path,
    observations: torch.Tensor,
    next_observations: torch.Tensor,
    rollout: EvalRollout,
    start_index: int,
    max_count: int,
    video_fps: int,
    sheet_timesteps: int,
    warmup_length: int,
) -> int:
    """Write MP4 and PNG diagnostics for samples from one batch."""

    end_index = min(max_count, start_index + observations.shape[0])
    for batch_index, output_index in enumerate(range(start_index, end_index)):
        ground_truth = next_observations[batch_index].detach().cpu().numpy()
        reconstruction = rollout.reconstructions[batch_index].detach().cpu().numpy()
        open_loop_prediction = rollout.open_loop_predictions[batch_index].detach().cpu().numpy()
        prefix = output_dir / f"sequence_{output_index:04d}"
        write_prediction_video(
            output_path=prefix.with_suffix(".mp4"),
            ground_truth=ground_truth,
            reconstruction=reconstruction,
            open_loop_prediction=open_loop_prediction,
            fps=video_fps,
            warmup_length=warmup_length,
        )
        write_prediction_contact_sheet(
            output_path=prefix.with_name(f"{prefix.name}_sheet.png"),
            ground_truth=ground_truth,
            reconstruction=reconstruction,
            open_loop_prediction=open_loop_prediction,
            timesteps=evenly_spaced_timesteps(ground_truth.shape[0], sheet_timesteps),
            warmup_length=warmup_length,
        )
    return end_index


def load_model_from_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_config: dict[str, Any],
    action_dim: int,
    image_size: int,
    device: torch.device,
) -> VisualWorldModel:
    """Instantiate and load a ``VisualWorldModel`` from a training checkpoint."""

    model = VisualWorldModel(
        action_dim=action_dim,
        embedding_size=int(checkpoint_config.get("embedding_size", 256)),
        hidden_size=int(checkpoint_config.get("hidden_size", 128)),
        latent_size=int(checkpoint_config.get("latent_size", 32)),
        image_size=image_size,
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    missing_latent_predictor = [name for name in missing if name.startswith("latent_predictor.")]
    disallowed_missing = [name for name in missing if not name.startswith("latent_predictor.")]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"checkpoint model state mismatch: missing={disallowed_missing}, unexpected={unexpected}",
        )
    setattr(model, "latent_predictor_loaded", not missing_latent_predictor)
    model.eval()
    return model


def select_sequence_indices(
    total_sequences: int,
    num_sequences: int | None,
    seed: int,
) -> list[int]:
    """Select deterministic dataset indices for evaluation."""

    if total_sequences < 1:
        raise ValueError("cannot evaluate an empty dataset")
    if num_sequences is None or num_sequences >= total_sequences:
        return list(range(total_sequences))
    if num_sequences < 1:
        raise ValueError("num_sequences must be >= 1 when set")
    rng = np.random.default_rng(seed)
    return sorted(int(index) for index in rng.choice(total_sequences, size=num_sequences, replace=False))


def resolve_sequence_length(config: EvalConfig, checkpoint_config: dict[str, Any]) -> int:
    """Use CLI sequence length when set; otherwise fall back to checkpoint config."""

    value = config.sequence_length or int(checkpoint_config.get("sequence_length", 16))
    if value < 1:
        raise ValueError("sequence_length must be >= 1")
    return value


def resolve_val_fraction(config: EvalConfig, checkpoint_config: dict[str, Any]) -> float:
    """Use CLI val fraction when set; otherwise fall back to checkpoint config."""

    value = config.val_fraction
    if value is None:
        value = float(checkpoint_config.get("val_fraction", 0.1))
    return value


def resolve_crop_config(config: EvalConfig, checkpoint_config: dict[str, Any]) -> ImageCropConfig:
    """Use CLI crop config when set; otherwise fall back to checkpoint config."""

    if config.crop_config is not None:
        return config.crop_config
    return crop_config_from_mapping(checkpoint_config.get("crop_config"))


def validate_eval_config(config: EvalConfig, sequence_length: int) -> None:
    """Validate evaluation settings before running model code."""

    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.warmup_length < 1:
        raise ValueError("warmup_length must be >= 1")
    if config.warmup_length >= sequence_length:
        raise ValueError("warmup_length must be smaller than sequence_length")
    if not config.horizons:
        raise ValueError("at least one prediction horizon is required")
    for horizon in config.horizons:
        if horizon < 1:
            raise ValueError("prediction horizons must be >= 1")
        if config.warmup_length + horizon - 1 > sequence_length:
            raise ValueError(
                f"horizon={horizon} exceeds sequence_length={sequence_length} "
                f"with warmup_length={config.warmup_length}",
            )
    if config.num_visualizations < 0:
        raise ValueError("num_visualizations must be >= 0")
    if config.video_fps < 1:
        raise ValueError("video_fps must be >= 1")
    if config.sheet_timesteps < 1:
        raise ValueError("sheet_timesteps must be >= 1")


def write_metrics(
    path: Path,
    metrics: dict[str, float],
    config: EvalConfig,
    selected_indices: Sequence[int],
    crop_config: ImageCropConfig,
) -> Path:
    """Write metrics and reproducibility metadata to JSON."""

    payload = {
        "metrics": metrics,
        "config": serializable_eval_config(config),
        "resolved_crop_config": asdict(crop_config),
        "selected_indices": list(selected_indices),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def serializable_eval_config(config: EvalConfig) -> dict[str, Any]:
    """Convert config dataclass values to JSON-friendly objects."""

    values = asdict(config)
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def print_metrics_table(console: Console, metrics: dict[str, float]) -> None:
    """Print a compact Rich table with evaluation metrics."""

    table = Table(title="RSSM Evaluation Metrics")
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    for key in sorted(metrics):
        table.add_row(key, f"{metrics[key]:.8f}")
    console.print(table)


def main() -> None:
    """Run the RSSM evaluation CLI."""

    console = Console()
    config = config_from_args(build_arg_parser().parse_args())
    console.rule("[bold cyan]RSSM evaluation config[/]")
    console.print_json(json.dumps(serializable_eval_config(config), indent=2))
    evaluate_checkpoint(config, console=console)


if __name__ == "__main__":
    main()
