"""Offline evaluation for the factored mechanics visual world model.

Mirrors the contract of ``eval-rssm`` so the two checkpoints can be compared
side-by-side: same JSON metric layout, same video / contact-sheet artifacts.
On top of those, this evaluator also runs the two non-training diagnostics
that are specific to the mechanics model:

* a held-out OLS linear probe from encoded ``q`` / ``qdot`` onto the
  simulator's stored ``physics_qpos`` / ``physics_qvel``, reporting R²
  per branch — the K2 factorization gate; and
* learned energy ``E = T(q, qdot) + V(q)`` over open-loop rollouts,
  reporting normalized drift — the secondary diagnostic for H3.

Probes are skipped automatically when the dataset has no ``physics_qpos``
fields (e.g. legacy collections), so legacy datasets still produce the
core MSE / video output.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.table import Table
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data import EpisodeSequenceDataset, ImageCropConfig
from data.image_transforms import (
    add_crop_args,
    crop_config_from_args,
    crop_config_from_mapping,
)
from eval.eval_rssm import select_sequence_indices
from eval.probes import (
    compute_learned_energy,
    fit_linear_probe,
    flatten_target,
    normalized_energy_drift,
)
from models.losses import foreground_reconstruction_mask
from models.mechanics import MechanicsWorldModel
from train.train_mechanics import resolve_device, set_seed
from viz.rssm_eval_viz import (
    evenly_spaced_timesteps,
    write_prediction_contact_sheet,
    write_prediction_video,
)


@dataclass(frozen=True)
class MechanicsEvalConfig:
    """Configuration for offline mechanics-model evaluation."""

    checkpoint_path: Path
    dataset_dir: Path
    output_dir: Path
    sequence_length: int | None = None
    batch_size: int = 16
    num_sequences: int | None = 64
    warmup_length: int = 5
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    seed: int = 0
    val_fraction: float | None = None
    split: str = "val"
    device: str = "auto"
    num_workers: int = 0
    num_visualizations: int = 3
    video_fps: int = 6
    sheet_timesteps: int = 8
    skip_probes: bool = False
    crop_config: ImageCropConfig | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--warmup-length", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-visualizations", type=int, default=3)
    parser.add_argument("--video-fps", type=int, default=6)
    parser.add_argument("--sheet-timesteps", type=int, default=8)
    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="disable the linear-probe and energy-tracking diagnostics",
    )
    add_crop_args(parser, default_mode=None)
    return parser


def config_from_args(args: argparse.Namespace) -> MechanicsEvalConfig:
    return MechanicsEvalConfig(
        checkpoint_path=args.checkpoint_path,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_sequences=args.num_sequences,
        warmup_length=args.warmup_length,
        horizons=tuple(sorted(set(args.horizons))),
        seed=args.seed,
        val_fraction=args.val_fraction,
        split=args.split,
        device=args.device,
        num_workers=args.num_workers,
        num_visualizations=args.num_visualizations,
        video_fps=args.video_fps,
        sheet_timesteps=args.sheet_timesteps,
        skip_probes=args.skip_probes,
        crop_config=crop_config_from_args(args),
    )


def evaluate_checkpoint(
    config: MechanicsEvalConfig, console: Console | None = None,
) -> dict[str, Any]:
    """Load a mechanics checkpoint, run all evaluators, and return a metrics dict."""

    console = console or Console()
    set_seed(config.seed)
    device = resolve_device(config.device)
    # weights_only=False: our checkpoints store NumPy RNG / scheduler state /
    # config dataclass next to the model weights — same rationale as in
    # train_mechanics.load_checkpoint.
    checkpoint = torch.load(
        config.checkpoint_path, map_location=device, weights_only=False,
    )
    if checkpoint.get("model_kind") not in (None, "mechanics"):
        raise RuntimeError(
            "checkpoint model_kind="
            f"{checkpoint.get('model_kind')!r} is not 'mechanics'; "
            "this evaluator is for MechanicsWorldModel only.",
        )
    checkpoint_config = dict(checkpoint.get("config", {}))
    sequence_length = _resolve_sequence_length(config, checkpoint_config)
    val_fraction = _resolve_val_fraction(config, checkpoint_config)
    crop_config = config.crop_config or crop_config_from_mapping(
        checkpoint_config.get("crop_config"),
    )
    _validate(config, sequence_length)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = EpisodeSequenceDataset(
        dataset_dir=config.dataset_dir,
        sequence_length=sequence_length,
        split=config.split,  # type: ignore[arg-type]
        val_fraction=val_fraction,
        seed=config.seed,
        crop_config=crop_config,
    )
    indices = select_sequence_indices(
        total_sequences=len(dataset),
        num_sequences=config.num_sequences,
        seed=config.seed,
    )
    eval_subset = Subset(dataset, indices)
    loader = DataLoader(
        eval_subset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, drop_last=False,
    )

    sample = dataset[indices[0]]
    action_dim = int(sample["actions"].shape[-1])
    image_size = int(sample["observations"].shape[-1])
    has_physics = "physics_qpos" in sample and "physics_qvel" in sample

    model = load_mechanics_model_from_checkpoint(
        checkpoint=checkpoint,
        checkpoint_config=checkpoint_config,
        action_dim=action_dim,
        image_size=image_size,
        device=device,
    )
    console.print(
        f"evaluating mechanics checkpoint={config.checkpoint_path} split={config.split} "
        f"sequences={len(eval_subset)} warmup={config.warmup_length} "
        f"horizons={list(config.horizons)} crop={crop_config} "
        f"physics_in_dataset={has_physics}",
    )

    rollouts = evaluate_loader(
        model=model,
        loader=loader,
        device=device,
        warmup_length=config.warmup_length,
        horizons=config.horizons,
        output_dir=config.output_dir,
        num_visualizations=config.num_visualizations,
        video_fps=config.video_fps,
        sheet_timesteps=config.sheet_timesteps,
        collect_factors=not config.skip_probes,
        collect_physics=not config.skip_probes and has_physics,
    )

    metrics: dict[str, Any] = dict(rollouts["metrics"])
    metrics.update(
        {
            "num_sequences": float(len(eval_subset)),
            "sequence_length": float(sequence_length),
            "warmup_length": float(config.warmup_length),
            "physics_in_dataset": bool(has_physics),
        },
    )

    if not config.skip_probes:
        if has_physics:
            metrics["linear_probe"] = run_linear_probes(
                rollouts["q"], rollouts["qdot"],
                rollouts["physics_qpos"], rollouts["physics_qvel"],
            )
        metrics["energy"] = run_energy_tracking(
            model=model,
            imagined_q=rollouts["imagined_q"],
            imagined_qdot=rollouts["imagined_qdot"],
            posterior_q=rollouts["posterior_q_warm"],
            posterior_qdot=rollouts["posterior_qdot_warm"],
            warmup_length=config.warmup_length,
        )

    write_metrics(config.output_dir / "metrics.json", metrics, config, indices, crop_config)
    print_metrics_table(console, metrics)
    return metrics


@torch.no_grad()
def evaluate_loader(
    model: MechanicsWorldModel,
    loader: DataLoader,
    device: torch.device,
    warmup_length: int,
    horizons: Sequence[int],
    output_dir: Path,
    num_visualizations: int,
    video_fps: int,
    sheet_timesteps: int,
    collect_factors: bool,
    collect_physics: bool,
) -> dict[str, Any]:
    """Iterate the loader, accumulate metrics, write visualizations, return tensors."""

    model.eval()
    recon_se = _MetricAcc()
    fg_recon_se = _MetricAcc()
    open_loop_se = _MetricAcc()
    fg_open_loop_se = _MetricAcc()
    per_horizon_se: dict[int, _MetricAcc] = {h: _MetricAcc() for h in horizons}
    per_horizon_fg_se: dict[int, _MetricAcc] = {h: _MetricAcc() for h in horizons}
    smoothness_se = _MetricAcc()

    factor_q: list[np.ndarray] = []
    factor_qdot: list[np.ndarray] = []
    physics_qpos: list[np.ndarray] = []
    physics_qvel: list[np.ndarray] = []
    imagined_q_chunks: list[np.ndarray] = []
    imagined_qdot_chunks: list[np.ndarray] = []
    posterior_q_warm_chunks: list[np.ndarray] = []
    posterior_qdot_warm_chunks: list[np.ndarray] = []

    saved = 0
    for batch in loader:
        observations = batch["observations"].to(device)
        next_observations = batch["next_observations"].to(device)
        actions = batch["actions"].to(device)
        sequence_length = observations.shape[1]
        max_horizon = sequence_length - warmup_length + 1

        # Forward without imagination — we just need the posterior + decoded
        # reconstructions for the same-step recon metric.
        outputs = model(observations=observations, actions=actions)
        recon_se.update(outputs.reconstructions, observations)
        fg_mask_obs = foreground_reconstruction_mask(observations).detach()
        fg_recon_se.update(outputs.reconstructions, observations, fg_mask_obs)

        # Open-loop imagination: warmup posterior for K_ctx steps, then roll
        # the integrator for the rest of the sequence. Aligned so that
        # predictions[:, k] targets next_observations[:, warmup_length - 1 + k]
        # / obs_{warmup_length + k}.
        with torch.enable_grad():
            (
                imagined_recon, _, imagined_q, imagined_qdot, _, target_start,
            ) = model.imagine_from_posterior(
                rollout=model._rollout(posterior=model.encoder(observations), actions=actions),
                actions=actions,
                context_steps=warmup_length,
                horizon=max_horizon,
            )
        imagined_recon = imagined_recon.detach()
        imagined_q = imagined_q.detach()
        imagined_qdot = imagined_qdot.detach()

        target = next_observations[:, target_start : target_start + max_horizon]
        fg_mask_pred = foreground_reconstruction_mask(target).detach()
        open_loop_se.update(imagined_recon, target)
        fg_open_loop_se.update(imagined_recon, target, fg_mask_pred)

        for horizon in horizons:
            if horizon - 1 >= imagined_recon.shape[1]:
                continue
            per_horizon_se[horizon].update(
                imagined_recon[:, horizon - 1], target[:, horizon - 1],
            )
            per_horizon_fg_se[horizon].update(
                imagined_recon[:, horizon - 1], target[:, horizon - 1],
                fg_mask_pred[:, horizon - 1],
            )

        # Smoothness diagnostic: how close encoded qdot is to the finite
        # difference of encoded q. Reported as the same MSE the trainer uses.
        posterior = model.encoder(observations)
        finite_diff = (posterior.q_mean[:, 1:] - posterior.q_mean[:, :-1]) / model.dt
        midpoint = 0.5 * (posterior.qdot_mean[:, 1:] + posterior.qdot_mean[:, :-1])
        smoothness_se.update(midpoint, finite_diff)

        if collect_factors:
            factor_q.append(_to_double_numpy(posterior.q_mean))
            factor_qdot.append(_to_double_numpy(posterior.qdot_mean))
            posterior_q_warm_chunks.append(
                _to_double_numpy(posterior.q_mean[:, warmup_length - 1 : warmup_length])
            )
            posterior_qdot_warm_chunks.append(
                _to_double_numpy(posterior.qdot_mean[:, warmup_length - 1 : warmup_length])
            )
            imagined_q_chunks.append(_to_double_numpy(imagined_q))
            imagined_qdot_chunks.append(_to_double_numpy(imagined_qdot))
        if collect_physics:
            physics_qpos.append(_to_double_numpy(batch["physics_qpos"]))
            physics_qvel.append(_to_double_numpy(batch["physics_qvel"]))

        if saved < num_visualizations:
            saved = _write_batch_visualizations(
                output_dir=output_dir,
                next_observations=next_observations,
                reconstructions=outputs.reconstructions,
                imagined=imagined_recon,
                target_start=target_start,
                start_index=saved,
                max_count=num_visualizations,
                video_fps=video_fps,
                sheet_timesteps=sheet_timesteps,
            )

    metrics = {
        "reconstruction_mse": recon_se.mean(),
        "foreground_reconstruction_mse": fg_recon_se.mean(),
        "open_loop_future_mse": open_loop_se.mean(),
        "foreground_open_loop_future_mse": fg_open_loop_se.mean(),
        "qdot_smoothness_mse": smoothness_se.mean(),
    }
    for horizon, acc in per_horizon_se.items():
        if acc.count > 0:
            metrics[f"open_loop_mse_h{horizon}"] = acc.mean()
            metrics[f"foreground_open_loop_mse_h{horizon}"] = per_horizon_fg_se[horizon].mean()

    return {
        "metrics": metrics,
        "q": np.concatenate(factor_q, axis=0) if factor_q else None,
        "qdot": np.concatenate(factor_qdot, axis=0) if factor_qdot else None,
        "physics_qpos": np.concatenate(physics_qpos, axis=0) if physics_qpos else None,
        "physics_qvel": np.concatenate(physics_qvel, axis=0) if physics_qvel else None,
        "imagined_q": (
            np.concatenate(imagined_q_chunks, axis=0) if imagined_q_chunks else None
        ),
        "imagined_qdot": (
            np.concatenate(imagined_qdot_chunks, axis=0) if imagined_qdot_chunks else None
        ),
        "posterior_q_warm": (
            np.concatenate(posterior_q_warm_chunks, axis=0)
            if posterior_q_warm_chunks else None
        ),
        "posterior_qdot_warm": (
            np.concatenate(posterior_qdot_warm_chunks, axis=0)
            if posterior_qdot_warm_chunks else None
        ),
    }


def run_linear_probes(
    encoded_q: np.ndarray,
    encoded_qdot: np.ndarray,
    physics_qpos: np.ndarray,
    physics_qvel: np.ndarray,
) -> dict[str, Any]:
    """OLS probe summary: encoded_q -> qpos and encoded_qdot -> qvel."""

    if encoded_q is None or physics_qpos is None:
        return {}
    flat_q = flatten_target(encoded_q)
    flat_qdot = flatten_target(encoded_qdot)
    flat_qpos = flatten_target(physics_qpos)
    flat_qvel = flatten_target(physics_qvel)
    q_probe = fit_linear_probe(flat_q, flat_qpos)
    qdot_probe = fit_linear_probe(flat_qdot, flat_qvel)
    return {
        "q_to_qpos_r2_overall": float(q_probe.r2_overall),
        "q_to_qpos_r2_per_dim": [float(x) for x in q_probe.r2_per_dim],
        "qdot_to_qvel_r2_overall": float(qdot_probe.r2_overall),
        "qdot_to_qvel_r2_per_dim": [float(x) for x in qdot_probe.r2_per_dim],
    }


def run_energy_tracking(
    model: MechanicsWorldModel,
    imagined_q: np.ndarray | None,
    imagined_qdot: np.ndarray | None,
    posterior_q: np.ndarray | None,
    posterior_qdot: np.ndarray | None,
    warmup_length: int,
) -> dict[str, Any]:
    """Compute learned-energy drift over the open-loop imagination window.

    ``E_t = T(q_t, qdot_t) + V(q_t)`` for the integrator's view of the system.
    We anchor the drift to the energy at the close-of-warmup (the last
    posterior step, which is where the open-loop chain starts), so a flat
    drift means the integrator is conserving the learned Hamiltonian.
    """

    if imagined_q is None or posterior_q is None:
        return {}
    if imagined_q.size == 0:
        return {}
    full_q = np.concatenate([posterior_q, imagined_q], axis=1)        # [B, 1+H, d]
    full_qdot = np.concatenate([posterior_qdot, imagined_qdot], axis=1)
    energy = compute_learned_energy(
        model,
        torch.from_numpy(full_q).float(),
        torch.from_numpy(full_qdot).float(),
    )
    drift = normalized_energy_drift(energy)
    abs_drift = np.abs(drift)
    return {
        "warmup_length": int(warmup_length),
        "horizons_evaluated": int(energy.shape[1]),
        "mean_abs_drift": float(abs_drift.mean()),
        "max_abs_drift": float(abs_drift.max()),
        "mean_abs_drift_h10": (
            float(abs_drift[:, min(10, abs_drift.shape[1] - 1)].mean())
            if abs_drift.shape[1] >= 2 else float("nan")
        ),
    }


def load_mechanics_model_from_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_config: dict[str, Any],
    action_dim: int,
    image_size: int,
    device: torch.device,
) -> MechanicsWorldModel:
    """Instantiate ``MechanicsWorldModel`` from a training checkpoint."""

    model = MechanicsWorldModel(
        action_dim=action_dim,
        q_dim=int(checkpoint_config.get("q_dim", 2)),
        nuisance_dim=int(checkpoint_config.get("nuisance_dim", 32)),
        embedding_size=int(checkpoint_config.get("embedding_size", 256)),
        encoder_head_hidden_size=int(
            checkpoint_config.get("encoder_head_hidden_size", 128),
        ),
        mass_matrix_hidden_size=int(
            checkpoint_config.get("mass_matrix_hidden_size", 64),
        ),
        potential_hidden_size=int(
            checkpoint_config.get("potential_hidden_size", 64),
        ),
        dissipation_hidden_size=int(
            checkpoint_config.get("dissipation_hidden_size", 64),
        ),
        actuation_hidden_size=int(
            checkpoint_config.get("actuation_hidden_size", 64),
        ),
        dynamics_hidden_layers=int(
            checkpoint_config.get("dynamics_hidden_layers", 2),
        ),
        image_size=image_size,
        dt=float(checkpoint_config.get("dt", 0.01)),
        n_substeps=int(checkpoint_config.get("n_substeps", 1)),
        with_dissipation=bool(checkpoint_config.get("with_dissipation", True)),
        nuisance_alpha_init=float(
            checkpoint_config.get("nuisance_alpha_init", 0.95),
        ),
        reward_head=bool(checkpoint_config.get("reward_head", False)),
        reward_head_hidden_size=int(
            checkpoint_config.get("reward_head_hidden_size", 200),
        ),
    ).to(device)
    optional_prefixes = ("reward_head.", "dissipation_module.")
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    disallowed_missing = [
        n for n in missing if not any(n.startswith(p) for p in optional_prefixes)
    ]
    unexpected_disallowed = [
        n for n in unexpected if not any(n.startswith(p) for p in optional_prefixes)
    ]
    if disallowed_missing or unexpected_disallowed:
        raise RuntimeError(
            "checkpoint model state mismatch: "
            f"missing={disallowed_missing}, unexpected={unexpected_disallowed}",
        )
    model.eval()
    return model


def _write_batch_visualizations(
    output_dir: Path,
    next_observations: torch.Tensor,
    reconstructions: torch.Tensor,
    imagined: torch.Tensor,
    target_start: int,
    start_index: int,
    max_count: int,
    video_fps: int,
    sheet_timesteps: int,
) -> int:
    """Write per-sequence MP4 + PNG artifacts for the post-warmup window only.

    The visualization slices each sequence to the imagination window
    (``next_observations[:, target_start : target_start + horizon]``) so the
    posterior-warmup phase is excluded — that's the part where reconstructions
    are nearly perfect anyway, so they would dominate the visual signal.
    """

    horizon = imagined.shape[1]
    end_index = min(max_count, start_index + next_observations.shape[0])
    for batch_idx, output_idx in enumerate(range(start_index, end_index)):
        ground_truth = (
            next_observations[batch_idx, target_start : target_start + horizon]
            .detach().cpu().numpy()
        )
        # Reconstruction strip aligned to the same window: posterior recon at
        # obs_{target_start + 1 .. target_start + horizon}.
        recon_window = reconstructions[
            batch_idx, target_start + 1 : target_start + 1 + horizon,
        ]
        if recon_window.shape[0] < horizon:
            # Final step has no obs_{T+1}; pad with the last available recon.
            pad = recon_window[-1:].repeat(horizon - recon_window.shape[0], 1, 1, 1)
            recon_window = torch.cat([recon_window, pad], dim=0)
        recon_arr = recon_window.detach().cpu().numpy()
        open_loop = imagined[batch_idx].detach().cpu().numpy()

        prefix = output_dir / f"sequence_{output_idx:04d}"
        write_prediction_video(
            output_path=prefix.with_suffix(".mp4"),
            ground_truth=ground_truth,
            reconstruction=recon_arr,
            open_loop_prediction=open_loop,
            fps=video_fps,
        )
        write_prediction_contact_sheet(
            output_path=prefix.with_name(f"{prefix.name}_sheet.png"),
            ground_truth=ground_truth,
            reconstruction=recon_arr,
            open_loop_prediction=open_loop,
            timesteps=evenly_spaced_timesteps(ground_truth.shape[0], sheet_timesteps),
        )
    return end_index


def write_metrics(
    output_path: Path,
    metrics: dict[str, Any],
    config: MechanicsEvalConfig,
    selected_indices: Sequence[int],
    crop_config: ImageCropConfig,
) -> Path:
    """Write a JSON summary mirroring the eval-rssm format."""

    payload = {
        "metrics": metrics,
        "config": _serializable_config(config),
        "selected_indices": list(selected_indices),
        "crop_config": asdict(crop_config),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def print_metrics_table(console: Console, metrics: dict[str, Any]) -> None:
    """Render a Rich table that summarizes the headline metrics."""

    table = Table(title="Mechanics Eval Metrics", show_header=False, box=None)
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("value", justify="right")
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            table.add_row(key, f"{value:.6f}")
        elif isinstance(value, int):
            table.add_row(key, str(value))
        elif isinstance(value, dict):
            for sub_key in sorted(value):
                sub_value = value[sub_key]
                if isinstance(sub_value, float):
                    table.add_row(f"{key}.{sub_key}", f"{sub_value:.6f}")
                elif isinstance(sub_value, list):
                    table.add_row(f"{key}.{sub_key}", str(sub_value))
                else:
                    table.add_row(f"{key}.{sub_key}", str(sub_value))
        else:
            table.add_row(key, str(value))
    console.print(table)


def _resolve_sequence_length(
    config: MechanicsEvalConfig, checkpoint_config: dict[str, Any],
) -> int:
    value = config.sequence_length or int(checkpoint_config.get("sequence_length", 32))
    if value < 1:
        raise ValueError("sequence_length must be >= 1")
    return value


def _resolve_val_fraction(
    config: MechanicsEvalConfig, checkpoint_config: dict[str, Any],
) -> float:
    value = config.val_fraction
    if value is None:
        value = float(checkpoint_config.get("val_fraction", 0.1))
    return value


def _validate(config: MechanicsEvalConfig, sequence_length: int) -> None:
    if config.warmup_length < 1:
        raise ValueError("warmup_length must be >= 1")
    if sequence_length <= config.warmup_length:
        raise ValueError(
            "sequence_length must be > warmup_length so the open-loop "
            f"window has at least one step; got T={sequence_length} "
            f"warmup={config.warmup_length}",
        )
    if any(h < 1 for h in config.horizons):
        raise ValueError("horizons must be positive integers")


def _serializable_config(config: MechanicsEvalConfig) -> dict[str, Any]:
    values = asdict(config)
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values


def _to_double_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().double().numpy()


class _MetricAcc:
    __slots__ = ("squared_error", "count")

    def __init__(self) -> None:
        self.squared_error = 0.0
        self.count = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError(f"shape mismatch: {prediction.shape} != {target.shape}")
        squared_error = (prediction - target) ** 2
        if weight is not None:
            squared_error = squared_error * weight
        self.squared_error += float(torch.sum(squared_error).detach().cpu())
        self.count += int(target.numel())

    def mean(self) -> float:
        if self.count == 0:
            raise ValueError("cannot average an empty metric accumulator")
        return self.squared_error / self.count


def main() -> None:
    console = Console()
    config = config_from_args(build_arg_parser().parse_args())
    console.rule("[bold cyan]mechanics eval config[/]")
    console.print_json(json.dumps(_serializable_config(config), indent=2))
    evaluate_checkpoint(config, console=console)


if __name__ == "__main__":
    main()
