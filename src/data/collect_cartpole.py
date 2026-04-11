"""CLI for collecting random cartpole-swingup pixel rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.trajectory import Trajectory, save_trajectory_npz


def _camera_id(value: str) -> int | str:
    """Parse numeric camera ids as ints while preserving named cameras."""

    try:
        return int(value)
    except ValueError:
        return value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command line parser for data collection."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-frames", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--env-name", type=str, default="cartpole-swingup")
    parser.add_argument("--camera-id", type=_camera_id, default=0)
    return parser


def collect_random_rollouts(
    output_dir: Path,
    total_frames: int,
    seed: int,
    action_repeat: int,
    image_size: int,
    env_name: str = "cartpole-swingup",
    camera_id: int | str = 0,
) -> list[Path]:
    """Collect random-policy transitions and save one NPZ file per episode."""

    from envs.dm_control_pixels import DmControlPixelEnv, PixelEnvConfig

    if total_frames < 1:
        raise ValueError("total_frames must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    config = PixelEnvConfig(
        env_name=env_name,
        seed=seed,
        action_repeat=action_repeat,
        image_size=image_size,
        camera_id=camera_id,
    )

    written_paths: list[Path] = []
    remaining = total_frames
    episode_id = 0

    with DmControlPixelEnv(config) as env:
        while remaining > 0:
            images = [env.reset()]
            actions = []
            rewards = []
            discounts = []
            dones = []
            step_indices = []

            local_step = 0
            done = False
            while remaining > 0 and not done:
                action = env.sample_action(rng)
                step = env.step(action)

                actions.append(np.asarray(action, dtype=np.float32))
                rewards.append(step.reward)
                discounts.append(step.discount)
                dones.append(step.done)
                step_indices.append(local_step)
                images.append(step.image)

                local_step += 1
                remaining -= 1
                done = step.done

            trajectory = Trajectory(
                env_name=env_name,
                seed=seed,
                episode_id=episode_id,
                action_repeat=action_repeat,
                image_size=image_size,
                images=np.stack(images, axis=0),
                actions=np.stack(actions, axis=0),
                rewards=np.asarray(rewards, dtype=np.float32),
                discounts=np.asarray(discounts, dtype=np.float32),
                dones=np.asarray(dones, dtype=np.bool_),
                step_indices=np.asarray(step_indices, dtype=np.int64),
            )
            path = output_dir / f"episode_{episode_id:06d}.npz"
            written_paths.append(save_trajectory_npz(trajectory, path))
            episode_id += 1

    return written_paths


def main() -> None:
    """Run the collection CLI."""

    args = build_arg_parser().parse_args()
    paths = collect_random_rollouts(
        output_dir=args.output_dir,
        total_frames=args.total_frames,
        seed=args.seed,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        env_name=args.env_name,
        camera_id=args.camera_id,
    )
    print(f"Wrote {len(paths)} trajectory file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
