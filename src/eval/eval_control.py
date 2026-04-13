"""Closed-loop MPC evaluation of a trained world model on dm_control envs.

Loads a checkpoint (must have been trained with ``--reward-head``), builds a
``CEMPlanner`` over the resulting model, and runs N MPC-controlled episodes
on the requested env (optionally perturbed by physics scales and visual
distractors). Writes a small ``returns.json`` + ``summary.csv`` so the
research-spec primary metric — normalized MPC return under shift — is
trivially read back.

This is the primary online-evaluation entry point. Offline metrics (pixel
prediction MSE, energy tracking, linear probes) live in ``eval-rssm`` /
``probes.py`` and stay independent.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console

from data.collect_cartpole import _physics_scale  # CLI parser reuse: KEY=SCALE
from envs.dm_control_pixels import DmControlPixelEnv, PixelEnvConfig
from envs.distractors import EvalPixelEnv, VisualDistractorConfig
from models import VisualWorldModel
from planning import CEMPlanner, plan_episode
from planning.cem import EpisodeRollout, aggregate_returns


@dataclass(frozen=True)
class EvalControlConfig:
    """All knobs for one ``eval-control`` invocation."""

    checkpoint_path: Path
    output_dir: Path
    env_name: str = "cartpole-swingup"
    image_size: int = 84
    action_repeat: int = 2
    seed: int = 0
    episodes: int = 10
    max_steps: int = 1000
    horizon: int = 15
    n_samples: int = 512
    n_iters: int = 3
    elite_frac: float = 0.125
    discount: float = 0.99
    device: str = "auto"
    physics_param_scales: dict[str, float] | None = field(default=None)
    color_gain_low: float = 1.0
    color_gain_high: float = 1.0
    brightness_low: float = 0.0
    brightness_high: float = 0.0
    nominal_return: float = 1000.0
    """Per-episode return achievable by an oracle. Used to normalize returns
    so cross-shift comparison is in the [0, 1] range expected by the spec."""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-name", type=str, default="cartpole-swingup")
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--n-iters", type=int, default=3)
    parser.add_argument("--elite-frac", type=float, default=0.125)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--physics-scale",
        action="append",
        type=_physics_scale,
        default=None,
        help=(
            "Multiplicative physics shift, e.g. --physics-scale body_mass_2=0.5. "
            "Repeat for multiple shifts. Applied freshly to every episode env."
        ),
    )
    parser.add_argument("--color-gain-low", type=float, default=1.0)
    parser.add_argument("--color-gain-high", type=float, default=1.0)
    parser.add_argument("--brightness-low", type=float, default=0.0)
    parser.add_argument("--brightness-high", type=float, default=0.0)
    parser.add_argument(
        "--nominal-return",
        type=float,
        default=1000.0,
        help=(
            "Per-episode return used as the denominator of the normalized "
            "metric. dm_control cartpole-swingup returns at most ~1000."
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> EvalControlConfig:
    physics_param_scales = (
        dict(args.physics_scale) if args.physics_scale else None
    )
    return EvalControlConfig(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        env_name=args.env_name,
        image_size=args.image_size,
        action_repeat=args.action_repeat,
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        horizon=args.horizon,
        n_samples=args.n_samples,
        n_iters=args.n_iters,
        elite_frac=args.elite_frac,
        discount=args.discount,
        device=args.device,
        physics_param_scales=physics_param_scales,
        color_gain_low=args.color_gain_low,
        color_gain_high=args.color_gain_high,
        brightness_low=args.brightness_low,
        brightness_high=args.brightness_high,
        nominal_return=args.nominal_return,
    )


def evaluate_control(
    config: EvalControlConfig, console: Console | None = None
) -> dict[str, Any]:
    """Run the MPC episodes described by ``config`` and write results."""

    console = console or Console()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config.device)

    raw_checkpoint = torch.load(
        config.checkpoint_path, map_location=device, weights_only=False
    )
    saved_train_config: dict[str, Any] = raw_checkpoint.get("config", {})
    if not saved_train_config.get("reward_head", False):
        raise RuntimeError(
            "MPC requires a checkpoint trained with --reward-head; this "
            "checkpoint reports reward_head=False in its config.",
        )

    # Build a probe env to query action spec — this is cheap and avoids
    # baking action_dim into the trainer's checkpoint payload.
    probe_env = DmControlPixelEnv(
        PixelEnvConfig(
            env_name=config.env_name,
            seed=config.seed,
            action_repeat=config.action_repeat,
            image_size=config.image_size,
        )
    )
    action_spec = probe_env.action_spec
    action_dim = int(np.prod(action_spec.shape))
    action_low = np.asarray(action_spec.minimum, dtype=np.float32)
    action_high = np.asarray(action_spec.maximum, dtype=np.float32)
    probe_env.close()

    model = VisualWorldModel(
        action_dim=action_dim,
        embedding_size=int(saved_train_config.get("embedding_size", 256)),
        hidden_size=int(saved_train_config.get("hidden_size", 128)),
        latent_size=int(saved_train_config.get("latent_size", 32)),
        image_size=config.image_size,
        reward_head=True,
        reward_head_hidden_size=int(saved_train_config.get("reward_head_hidden_size", 200)),
    ).to(device)
    missing, unexpected = model.load_state_dict(raw_checkpoint["model_state"], strict=False)
    if missing or unexpected:
        # The reward head must be present on the checkpoint.
        critical = [name for name in missing if not name.startswith("latent_predictor.")]
        if critical:
            raise RuntimeError(
                f"checkpoint missing required parameters: {critical}",
            )
    model.eval()

    planner = CEMPlanner(
        model=model,
        horizon=config.horizon,
        action_low=action_low,
        action_high=action_high,
        n_samples=config.n_samples,
        n_iters=config.n_iters,
        elite_frac=config.elite_frac,
        discount=config.discount,
        device=device,
        seed=config.seed,
    )

    distractor_active = (
        config.color_gain_high > config.color_gain_low
        or config.color_gain_low != 1.0
        or config.brightness_high > config.brightness_low
        or config.brightness_low != 0.0
    )
    distractor_config = VisualDistractorConfig(
        color_gain_low=config.color_gain_low,
        color_gain_high=config.color_gain_high,
        brightness_low=config.brightness_low,
        brightness_high=config.brightness_high,
        seed=config.seed,
    )

    rollouts: list[EpisodeRollout] = []
    for episode_index in range(config.episodes):
        env_config = PixelEnvConfig(
            env_name=config.env_name,
            seed=config.seed + episode_index,
            action_repeat=config.action_repeat,
            image_size=config.image_size,
            physics_param_scales=config.physics_param_scales,
        )
        with DmControlPixelEnv(env_config) as base_env:
            env: Any = (
                EvalPixelEnv(base_env, distractor_config)
                if distractor_active
                else base_env
            )
            rollout = plan_episode(
                env=env,
                planner=planner,
                max_steps=config.max_steps,
                record_images=False,
            )
            rollouts.append(rollout)
            console.print(
                f"[bold cyan]episode {episode_index + 1}/{config.episodes}[/]: "
                f"return={rollout.total_reward:.2f} length={rollout.length} "
                f"done={rollout.done}",
            )

    aggregate = aggregate_returns(rollouts)
    aggregate["normalized_return_mean"] = (
        aggregate["return_mean"] / config.nominal_return
        if config.nominal_return > 0.0
        else 0.0
    )

    write_results(config, rollouts, aggregate)
    console.print(
        f"[bold green]done[/] — return_mean={aggregate['return_mean']:.2f} "
        f"normalized={aggregate['normalized_return_mean']:.3f}",
    )
    return aggregate


def write_results(
    config: EvalControlConfig,
    rollouts: list[EpisodeRollout],
    aggregate: dict[str, Any],
) -> None:
    """Persist per-episode and aggregate results in machine-readable form."""

    output_dir = config.output_dir

    payload = {
        "config": _serialize_config(config),
        "aggregate": aggregate,
        "episodes": [
            {
                "episode_index": index,
                "return": rollout.total_reward,
                "length": rollout.length,
                "done": rollout.done,
                "elite_score_means": rollout.elite_score_means,
                "elite_score_stds": rollout.elite_score_stds,
            }
            for index, rollout in enumerate(rollouts)
        ],
    }
    (output_dir / "returns.json").write_text(json.dumps(payload, indent=2))

    csv_rows = [
        {
            "episode_index": index,
            "return": rollout.total_reward,
            "length": rollout.length,
            "done": int(rollout.done),
        }
        for index, rollout in enumerate(rollouts)
    ]
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_index", "return", "length", "done"])
        writer.writeheader()
        writer.writerows(csv_rows)


def _serialize_config(config: EvalControlConfig) -> dict[str, Any]:
    raw = asdict(config)
    for key, value in raw.items():
        if isinstance(value, Path):
            raw[key] = str(value)
    return raw


def _resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    """Run the MPC eval CLI."""

    console = Console()
    config = config_from_args(build_arg_parser().parse_args())
    console.rule("[bold cyan]eval-control config[/]")
    console.print_json(json.dumps(_serialize_config(config), indent=2))
    evaluate_control(config, console=console)


if __name__ == "__main__":
    main()
