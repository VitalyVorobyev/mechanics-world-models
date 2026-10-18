"""Print the progress of the gate pipeline (scripts/gate_pipeline.sh status).

Reads only files the pipeline and trainer write — runs/gate/*, each arm's
checkpoints/gate-<arm>/progress.json, and the OOD dataset directories — so
it is safe to run at any time, including while training.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

RUN_DIR = Path("runs/gate")
ARMS = ("lagrangian", "mlp")
REGIMES = ("mass07", "mass13", "length07", "length13", "mass20", "length15")
EXPECTED_EPISODES = 30  # 15k frames / 500 steps per episode
STAGES = (
    [f"collect-{r}" for r in REGIMES]
    + ["train-lagrangian", "train-mlp", "eval", "report"]
)
# Offline fg open-loop MSE of rssm-phase0-reward on the nominal val split
# (256 windows, warmup 5) — the bar for gate criterion 2.
RSSM_FG_H20 = 0.00917


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    return f"{hours}h{rem // 60:02d}m"


def _pipeline_pid() -> int | None:
    pid_file = RUN_DIR / "pipeline.pid"
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
    except (ValueError, ProcessLookupError, PermissionError):
        return None
    return pid


def _arm_line(arm: str) -> list[str]:
    path = Path(f"checkpoints/gate-{arm}/progress.json")
    if not path.exists():
        return [f"  {arm:<11} not started"]
    p = json.loads(path.read_text())
    rate = p.get("steps_per_second_this_session")
    age = time.time() - p.get("updated_unix", time.time())
    lines = [
        f"  {arm:<11} {p['status']:<9} epoch {p['epoch']}/{p['total_epochs']}  "
        f"step {p['global_step']}/{p['total_steps']} "
        f"({100 * (p.get('fraction_done') or 0):.1f}%)  "
        f"{(1 / rate if rate else float('nan')):.2f} s/step  "
        f"ETA {_fmt_duration(p.get('eta_seconds'))}  (updated {_fmt_duration(age)} ago)",
    ]
    train = p.get("latest_train") or {}
    if train:
        lines.append(
            "              train: "
            f"total {train.get('total_loss', float('nan')):.4f}  "
            f"recon {train.get('reconstruction_loss', float('nan')):.5f}  "
            f"kl q/qdot/z {train.get('kl_q_raw', float('nan')):.2f}/"
            f"{train.get('kl_qdot_raw', float('nan')):.2f}/"
            f"{train.get('kl_z_raw', float('nan')):.2f}  "
            f"reward {train.get('reward_loss', float('nan')):.4f}  "
            f"skipped {train.get('step_skipped', 0):.0f}",
        )
    probe = p.get("latest_val_open_loop") or {}
    if probe:
        h20 = probe.get("val_open_loop_fg_mse_h20")
        mark = "" if h20 is None else ("  (<= RSSM bar)" if h20 <= RSSM_FG_H20 else "  (> RSSM bar)")
        lines.append(
            "              live fg MSE h1/h5/h10/h20: "
            + "/".join(
                f"{probe.get(f'val_open_loop_fg_mse_h{h}', float('nan')):.4f}"
                for h in (1, 5, 10, 20)
            )
            + f"  [RSSM offline h20 {RSSM_FG_H20}]{mark}",
        )
    return lines


def main() -> None:
    pid = _pipeline_pid()
    print(f"gate pipeline: {'RUNNING (pid %d)' % pid if pid else 'not running'}")
    print("stages:")
    current_marked = False
    for stage in STAGES:
        done = (RUN_DIR / f"{stage}.done").exists()
        if done:
            state = "done"
        elif pid and not current_marked:
            state, current_marked = "RUNNING", True
        else:
            state = "pending"
        extra = ""
        if stage.startswith("collect-"):
            regime = stage.removeprefix("collect-")
            out = Path(f"data/cartpole-ood-{regime}-15k-v2")
            count = len(list(out.glob("episode_*.npz"))) if out.exists() else 0
            extra = f"  {count}/{EXPECTED_EPISODES} episodes"
        print(f"  {stage:<17} {state}{extra}")
    print("training:")
    for arm in ARMS:
        print("\n".join(_arm_line(arm)))
    report = RUN_DIR / "report.md"
    if report.exists():
        print(f"report: {report}")
    log = RUN_DIR / "pipeline.log"
    if log.exists():
        print("recent pipeline log:")
        for line in log.read_text().splitlines()[-4:]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
