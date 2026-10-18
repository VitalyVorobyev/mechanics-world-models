"""Rung 3 matched baseline CLI: context-conditioned MLP vs Lagrangian slots (H1b).

Both forms train on simulated cart masses (default 1.0 nominal, 0.7, 0.85,
1.2, 1.4; state only, generated in seconds) with a per-mass 3-number
context, then adapt only a new context to cart mass x0.5 and x2, which lie
outside the training range. ``--param actuator_gear_0`` (or ``gravity``)
varies that parameter instead, for both training and ``--sim-shift``.

Usage:
  .venv/bin/python scripts/oracles/context_oracle.py --seed 0
  .venv/bin/python scripts/oracles/context_oracle.py --param actuator_gear_0 --sim-shift gear_x2=2

Writes runs/oracles/context_oracle/<timestamp>.json. Resumable through
--cache-dir like dynamics_oracle.py; progress lines "[cells done/total]".
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import torch

from oracles.common import StateEpisodes, load_shift_splits, load_state_episodes
from oracles.context import CONTEXT_MODES, run_context_oracle, simulate_state_episodes

# Environment-name prefix per varied parameter (cart mass keeps its old names so caches stay valid).
PARAM_PREFIX = {"body_mass_1": "cart_mass", "actuator_gear_0": "gear", "gravity": "gravity"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nominal", type=Path, default=Path("data/cartpole-swingup-random-100k-v2"))
    parser.add_argument(
        "--shift", action="append", default=None, help="NAME=DATASET_DIR (default: cart mass x0.5 and x2)",
    )
    parser.add_argument(
        "--sim-shift", action="append", default=None,
        help="NAME=SCALE of --param: simulate a test shift (30 episodes split 15/7/8) instead of datasets",
    )
    parser.add_argument(
        "--param", choices=sorted(PARAM_PREFIX), default="body_mass_1",
        help="physics scale varied across training environments and simulated shifts (default: cart mass)",
    )
    parser.add_argument(
        "--train-scales", "--train-masses", dest="train_scales", type=float, nargs="+",
        default=[1.0, 0.7, 0.85, 1.2, 1.4],
    )
    parser.add_argument("--episodes-per-mass", type=int, default=30)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 20])
    parser.add_argument("--pretrain-steps", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=list(CONTEXT_MODES))
    parser.add_argument(
        "--pose-noise", type=float, nargs=2, default=[0.0, 0.0], metavar=("SIGMA_X_M", "SIGMA_THETA_DEG"),
        help="train/adapt/select on noisy poses with finite-difference velocities (pixel-encoder stand-in)",
    )
    parser.add_argument(
        "--start-context", type=int, default=0,
        help="with --pose-noise: compare start-state estimators using this many context frames (e.g. 8)",
    )
    parser.add_argument(
        "--train-velocity", choices=("central", "savgol"), default="central",
        help="with --pose-noise: velocity stencil for training/adaptation/selection windows",
    )
    parser.add_argument(
        "--encoder", type=Path, default=None,
        help="pose encoder saved by encoder_oracle.py --save: train/adapt on poses read from rendered frames",
    )
    parser.add_argument("--encoder-device", default="mps")
    parser.add_argument(
        "--debias", action="store_true",
        help="with --encoder: remove the encoder's per-pose median error, fitted on the nominal training "
        "episodes with true poses (an oracle calibration, diagnostic only)",
    )
    parser.add_argument(
        "--all-frames", action="store_true", help="with --encoder: keep frames where the cart or pole tip is out of view",
    )
    parser.add_argument(
        "--potential-slot", action="store_true",
        help="4-number contexts; the Lagrangian's 4th slot scales the potential (for gravity shifts)",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/oracles/context_oracle/cache"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.train_scales[0] != 1.0:
        parser.error("the first training scale must be the nominal 1.0 (its context is pinned to zero)")
    prefix = PARAM_PREFIX[args.param]

    shift_items = args.shift or [
        "cart_mass_x05=data/cartpole-ood-cartmass05-15k-v2",
        "cart_mass_x2=data/cartpole-ood-cartmass20-15k-v2",
    ]
    train_envs = {
        f"{prefix}_x{m:g}": simulate_state_episodes(
            None if m == 1.0 else {args.param: m}, episodes=args.episodes_per_mass, seed=1000 * args.seed + i,
        )
        for i, m in enumerate(args.train_scales)
    }
    nominal_val = load_state_episodes(args.nominal, split="val", seed=args.seed)
    shifted, shift_scales = {}, {}
    if args.sim_shift:
        for index, item in enumerate(args.sim_shift):
            name, _, scale = item.partition("=")
            shift_scales[name] = {args.param: float(scale)}
            sim = simulate_state_episodes(shift_scales[name], episodes=30, seed=1000 * args.seed + 500 + index)
            shifted[name] = tuple(sim.select(part) for part in (slice(0, 15), slice(15, 22), slice(22, 30)))
    else:
        for item in shift_items:
            name, _, directory = item.partition("=")
            shifted[name] = load_shift_splits(Path(directory), seed=args.seed)
    pose_source = None
    if args.encoder is not None:
        from oracles.encoder import load_pose_encoder
        from oracles.pixel_pose import with_encoder_poses

        digest = hashlib.sha256(args.encoder.read_bytes()).hexdigest()[:10]
        pose_source = f"encoder:{digest}:{'all' if args.all_frames else 'visible'}"
        encoder = load_pose_encoder(args.encoder, args.encoder_device)
        encode = lambda e, scales: with_encoder_poses(  # noqa: E731
            e, scales, encoder, device=args.encoder_device, visible_only=not args.all_frames,
        )
        train_envs = {
            name: encode(e, None if m == 1.0 else {args.param: m})
            for (name, e), m in zip(train_envs.items(), args.train_scales)
        }
        shifted = {name: tuple(encode(e, shift_scales.get(name)) for e in parts) for name, parts in shifted.items()}
        if args.debias:
            from oracles.pixel_pose import PoseCalibration

            calibration = PoseCalibration().fit(next(iter(train_envs.values())))
            train_envs = {name: calibration.apply(e) for name, e in train_envs.items()}
            shifted = {name: tuple(calibration.apply(e) for e in parts) for name, parts in shifted.items()}
            pose_source += ":debiased"
        print(f"poses from {args.encoder} ({pose_source})", flush=True)
    result = run_context_oracle(
        train_envs, nominal_val, shifted, ks=args.ks, pretrain_steps=args.pretrain_steps, seed=args.seed,
        modes=args.modes, pose_noise=(args.pose_noise[0], math.radians(args.pose_noise[1])),
        shift_scales=shift_scales or None, start_context=args.start_context,
        train_velocity=args.train_velocity, pose_source=pose_source,
        potential_slot=args.potential_slot, cache_dir=args.cache_dir, log=lambda msg: print(msg, flush=True),
    )
    path = result.write()
    print(result.summary())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
