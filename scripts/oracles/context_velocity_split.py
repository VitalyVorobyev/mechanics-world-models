"""Context-oracle diagnostic: are adaptation errors concentrated at unseen speeds?

Re-adapts ``lag_ctx`` and ``mlp_ctx`` (k segments, each mode's selected
recipe from a finished context-oracle run) and splits the shifted test
windows by whether their speeds over the first 20 frames exceed the 99th
percentile of the training data. If the MLP's excess error disappears
inside the training speeds, the advantage is state extrapolation; if it
remains, it is parameter extrapolation.

Usage:
  .venv/bin/python scripts/oracles/context_velocity_split.py --seed 0 \
      --shift-dir data/cartpole-ood-cartmass05-15k-v2 --shift-name cart_mass_x05
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from oracles.common import load_shift_splits
from oracles.context import ContextDynamics, _Adapted, _multi_env_windows, simulate_state_episodes
from oracles.dynamics import _pose, _subset, _windows, fit, free_segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shift-dir", type=Path, default=Path("data/cartpole-ood-cartmass05-15k-v2"))
    parser.add_argument("--shift-name", default="cart_mass_x05")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/oracles/context_oracle/cache"))
    args = parser.parse_args()
    torch.set_num_threads(4)

    runs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob("runs/oracles/context_oracle/2026*.json"))]
    run = next(
        r for r in reversed(runs) if r["config"]["seed"] == args.seed and args.shift_name in r["config"]["shifts"]
    )
    config, recipes = run["config"], run["config"]["recipes"]
    pretrained = next(
        Path(c).parent for c in glob.glob(str(args.cache_dir / "pretrained" / "*" / "config.json"))
        if json.loads(Path(c).read_text())["seed"] == args.seed
    )
    masses = [1.0 if name == "cart_mass_x1" else float(name.removeprefix("cart_mass_x")) for name in config["train_envs"]]
    episodes = config["train_episodes"][config["train_envs"][0]]
    train = [
        simulate_state_episodes(None if m == 1.0 else {"body_mass_1": m}, episodes=episodes, seed=1000 * args.seed + i)
        for i, m in enumerate(masses)
    ]
    p99 = np.percentile(np.abs(np.concatenate([t.qvel.reshape(-1, 2) for t in train])), 99, axis=0)

    shift_train, _, test = load_shift_splits(args.shift_dir, seed=args.seed)
    horizon, stride, margin = max(config["eval_horizons"]), config["eval_stride"], config["rail_margin"]
    starts = np.arange(0, test.qpos.shape[1] - horizon, stride)
    reach = np.stack([np.abs(test.qpos[:, s : s + horizon + 1, 0]).max(1) for s in starts], 1).reshape(-1)
    keep = reach < 1.8 - margin
    speed = np.stack([np.abs(test.qvel[:, s : s + 21]).max(1) for s in starts], 1).reshape(-1, 2)[keep]
    outside = (speed > p99).any(-1)
    windows = _windows(test, horizon, stride=stride, rail_margin=margin)
    segments = free_segments(shift_train, config["segment_length"], margin, args.seed)
    data = _windows(_subset(shift_train, segments[: args.k], config["segment_length"]), config["train_horizon"],
                    rail_margin=margin)
    print(f"train p99 |xdot|, |thetadot|: {p99.round(2)}; {keep.sum()} test windows, {outside.mean():.2f} outside")
    for form, mode in (("lagrangian", "lag_ctx"), ("mlp", "mlp_ctx")):
        model = ContextDynamics(form, len(masses), n_substeps=config["n_substeps"])
        model.load_state_dict(torch.load(pretrained / f"{form}.pt", weights_only=True))
        model.calibrate_slots(_multi_env_windows([train[0]], config["train_horizon"], margin)[0])
        lr, steps = recipes[mode].split("@")
        fit(_Adapted(model), data, model.adaptation_parameters(mode), steps=int(steps), lr=float(lr),
            batch_size=config["batch_size"], seed=args.seed + args.k)
        pred = model.rollout(windows[0], windows[1], windows[2]).detach()
        dev = (_pose(pred) - _pose(windows[3])).abs().mean(-1)[:, 19].numpy()
        print(f"{mode}: h20 all {dev.mean():.4f}  inside {dev[~outside].mean():.4f}  outside {dev[outside].mean():.4f}")


if __name__ == "__main__":
    main()
