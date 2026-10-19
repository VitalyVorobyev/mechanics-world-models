"""Export real rollouts of the H1b context oracle for the animated web demo.

For each scenario (a finished true-state ``context_oracle`` run, named by its
result JSON), this loads the cached pretrained Lagrangian and context MLP,
re-runs the selected one-clip (k=1) context adaptation for every shift, and
records open-loop rollouts against MuJoCo ground truth:

* the 50-frame adaptation clip each model saw;
* sampled long windows (``--horizon`` frames, from the true start state, the
  recorded random actions) for true / adapted / unadapted models;
* the mean error curve over all rail-free test windows, and the h20 errors on
  the oracle's own windows (a check that the run is reproduced).

It also renders one nominal episode and reads it with the label-free colour
heuristic and the keypoint encoder (the pixel front-end panel).

No training: a scenario whose pretrained weights are not cached is refused.

Usage:
  .venv/bin/python scripts/oracles/export_demo.py --out runs/demo
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from oracles.ceiling import SLIDER_LIMIT
from oracles.context import ContextDynamics, _Adapted, simulate_state_episodes
from oracles.dynamics import _config_hash, _pose, _subset, _windows, fit, free_segments, windows_error

TRAIN_SCALES = (1.0, 0.7, 0.85, 1.2, 1.4)
SCENARIOS = {
    "mass": {
        "title": "Cart mass",
        "param": "body_mass_1",
        "prefix": "cart_mass",
        "result": "20261018-193931.json",
        "potential_slot": False,
    },
    "gravity_noslot": {
        "title": "Gravity, no matching slot",
        "param": "gravity",
        "prefix": "gravity",
        "result": "20261018-233549.json",
        "potential_slot": False,
    },
    "gravity_slot": {
        "title": "Gravity, with a potential slot",
        "param": "gravity",
        "prefix": "gravity",
        "result": "20261019-000539.json",
        "potential_slot": True,
    },
}


def _rollout(model: ContextDynamics, q0, v0, act) -> np.ndarray:
    with torch.enable_grad():  # the Lagrangian step uses autograd internally
        return _Adapted(model).rollout(q0, v0, act).detach().double().numpy()


def _error_curve(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean |Δ(x, cos, sin)| per horizon step, as in the oracle."""

    t = torch.as_tensor
    return (_pose(t(pred)) - _pose(t(target))).abs().mean(dim=(0, 2)).numpy()


def _load_models(spec: dict, seed: int, train_envs, cache_dir: Path, frame_dt: float) -> dict[str, ContextDynamics]:
    names = list(train_envs)
    pretrain_config = {
        "oracle": "context",
        "train_envs": names,
        "train_episodes": {n: int(e.qpos.shape[0]) for n, e in train_envs.items()},
        "train_horizon": 8,
        "pretrain_steps": 15000,
        "batch_size": 128,
        "lr": 1e-3,
        "n_substeps": 2,
        "rail_margin": 0.2,
        "actuated_dims": [0],
        "seed": seed,
        "frame_dt": frame_dt,
    }
    if spec["potential_slot"]:
        pretrain_config["potential_slot"] = True
    root = cache_dir / "pretrained" / _config_hash(pretrain_config)
    calib = _windows(next(iter(train_envs.values())), 8, rail_margin=0.2)[0]
    models = {}
    for form in ("lagrangian", "mlp"):
        path = root / f"{form}.pt"
        if not path.exists():
            raise FileNotFoundError(f"no cached pretrained {form} at {path}; run the context oracle first")
        model = ContextDynamics(form, len(names), dt=frame_dt, n_substeps=2, potential_slot=spec["potential_slot"])
        model.load_state_dict(torch.load(path, weights_only=True))
        model.calibrate_slots(calib)
        models[form] = model
    return models


def export_scenario(spec: dict, *, horizon: int, samples: int, cache_dir: Path, results_dir: Path, log) -> dict:
    result = json.loads((results_dir / spec["result"]).read_text())
    config, metrics = result["config"], result["metrics"]
    seed, recipes = config["seed"], config["recipes"]
    train_envs = {
        f"{spec['prefix']}_x{m:g}": simulate_state_episodes(
            None if m == 1.0 else {spec["param"]: m}, episodes=30, seed=1000 * seed + i,
        )
        for i, m in enumerate(TRAIN_SCALES)
    }
    assert list(train_envs) == config["train_envs"], (list(train_envs), config["train_envs"])
    frame_dt = config["frame_dt"]
    models = _load_models(spec, seed, train_envs, cache_dir, frame_dt)
    out = {"title": spec["title"], "result": spec["result"], "seed": seed, "recipes": recipes, "shifts": []}
    for index, name in enumerate(config["shifts"]):
        start = time.time()
        scale = float(name.rsplit("_x", 1)[1])
        sim = simulate_state_episodes({spec["param"]: scale}, episodes=30, seed=1000 * seed + 500 + index)
        shift_train, shift_test = sim.select(slice(0, 15)), sim.select(slice(22, 30))
        segments = free_segments(shift_train, 50, 0.2, seed)
        clip = _subset(shift_train, segments[:1], 50)
        adapt_data = _windows(clip, 8, rail_margin=0.2, rail_reference=clip.qpos)
        adapted = {}
        for mode, form in (("lag_ctx", "lagrangian"), ("mlp_ctx", "mlp")):
            model = copy.deepcopy(models[form])
            lr, steps = recipes[mode].split("@")
            if int(steps) > 0:
                fit(
                    _Adapted(model), adapt_data, model.adaptation_parameters(mode), steps=int(steps), lr=float(lr),
                    batch_size=128, seed=seed + 1,
                )
            adapted[form] = model
        # Reproduction check on the oracle's own test windows.
        check = _windows(shift_test, 50, stride=5, rail_margin=0.2)
        h20 = {form: windows_error(_Adapted(m), check, (20,))[20] for form, m in adapted.items()}
        reported = {
            "lagrangian": metrics[f"{name}.k1.lag_ctx.dev_h20"], "mlp": metrics[f"{name}.k1.mlp_ctx.dev_h20"],
        }
        # Long windows for the animation and the error curves.
        q0, v0, act, tgt = _windows(shift_test, horizon, stride=10, rail_margin=0.2)
        preds = {f"{form}": _rollout(m, q0, v0, act) for form, m in adapted.items()}
        preds.update({f"{form}_unadapted": _rollout(m, q0, v0, act) for form, m in models.items()})
        target = tgt.double().numpy()
        curves = {k: _error_curve(v, target).round(5).tolist() for k, v in preds.items()}
        pick = np.random.default_rng(seed).choice(q0.shape[0], size=min(samples, q0.shape[0]), replace=False)
        windows = []
        for i in sorted(pick.tolist()):
            true_q = np.concatenate([q0[i : i + 1].double().numpy(), target[i]])
            windows.append({
                "actions": act[i, :, 0].numpy().round(3).tolist(),
                "true": true_q.round(4).tolist(),
                **{k: np.concatenate([true_q[:1], v[i]]).round(4).tolist() for k, v in preds.items()},
            })
        contexts = {form: m.adapted.detach().numpy().round(4).tolist() for form, m in adapted.items()}
        out["shifts"].append({
            "name": name,
            "scale": scale,
            "in_training_range": TRAIN_SCALES[1] <= scale <= TRAIN_SCALES[-1] or scale == 1.0,
            "clip": clip.qpos[0].round(4).tolist(),
            "clip_actions": clip.actions[0, :, 0].round(3).tolist(),
            "h20": h20,
            "h20_reported": reported,
            "contexts": contexts,
            "n_windows": int(q0.shape[0]),
            "curves": curves,
            "windows": windows,
        })
        log(
            f"{spec['title']} {name}: h20 lag {h20['lagrangian']:.4f} (reported {reported['lagrangian']:.4f}) "
            f"mlp {h20['mlp']:.4f} (reported {reported['mlp']:.4f}); {q0.shape[0]} windows of {horizon} "
            f"({time.time() - start:.0f}s)",
        )
    return out


def export_front_end(frames: int, log) -> dict:
    """One nominal episode rendered, read by the colour heuristic and the keypoint encoder."""

    from PIL import Image

    from oracles.bootstrap import heuristic_poses
    from oracles.encoder import encode_poses, in_view, load_pose_encoder
    from oracles.pixel_pose import render_states

    episodes = simulate_state_episodes(None, episodes=1, seed=4242)
    qpos = episodes.qpos[0, :frames]
    images = render_states(qpos, None)
    heuristic, found = heuristic_poses(images)
    encoder = load_pose_encoder(Path("runs/oracles/encoders/keypoint-cosine20k-150ep.pt"), "cpu")
    keypoint = encode_poses(encoder, images, "cpu")
    cols = 20
    rows = -(-frames // cols)
    sheet = np.zeros((rows * 84, cols * 84, 3), dtype=np.uint8)
    for i, image in enumerate(images):
        r, c = divmod(i, cols)
        sheet[r * 84 : (r + 1) * 84, c * 84 : (c + 1) * 84] = image
    buffer = io.BytesIO()
    Image.fromarray(sheet).save(buffer, format="PNG", optimize=True)
    wrap = lambda a: np.angle(np.exp(1j * a))  # noqa: E731
    log(f"front end: {frames} frames, sprite {len(buffer.getvalue()) / 1e3:.0f} kB")
    return {
        "frames": frames,
        "cols": cols,
        "sprite": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
        "true": np.stack([qpos[:, 0], wrap(qpos[:, 1])], -1).round(4).tolist(),
        "heuristic": np.stack([heuristic[:, 0], wrap(heuristic[:, 1])], -1).round(4).tolist(),
        "heuristic_found": found.tolist(),
        "keypoint": keypoint.round(4).tolist(),
        "in_view": in_view(qpos).tolist(),
        "pixels_per_metre": 25.35,
        "rail": SLIDER_LIMIT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("runs/demo"))
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--horizon", type=int, default=100, help="frames per animated window (0.02 s each)")
    parser.add_argument("--samples", type=int, default=6, help="animated windows per shift")
    parser.add_argument("--front-end-frames", type=int, default=200)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/oracles/context_oracle/cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("runs/oracles/context_oracle"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    log = lambda msg: print(msg, flush=True)  # noqa: E731
    for key in args.scenarios:
        data = export_scenario(
            SCENARIOS[key], horizon=args.horizon, samples=args.samples, cache_dir=args.cache_dir,
            results_dir=args.results_dir, log=log,
        )
        (args.out / f"{key}.json").write_text(json.dumps(data))
    if args.front_end_frames > 0:
        (args.out / "front_end.json").write_text(json.dumps(export_front_end(args.front_end_frames, log)))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
