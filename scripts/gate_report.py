"""Summarize eval/gate/* into a Markdown report with the gate verdict.

Usage: .venv/bin/python scripts/gate_report.py [--output runs/gate/report.md]

Reads the metrics.json files written by scripts/eval_gate.sh for the
``lagrangian`` and ``mlp`` gate arms and the ``rssm`` baseline, and applies
the criteria from docs/current_status.md ("Gate"):

1. gate-lagrangian ``state_probes.q->pose.mlp`` R² > 0.8 (nominal val split);
2. its nominal fg open-loop MSE at h20 <= the RSSM's;
3. its mean OOD degradation (fg h20 OOD / fg h20 nominal) < the RSSM's;
and gate-lagrangian must beat gate-mlp on 2 and 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EVAL_ROOT = Path("eval/gate")
MODELS = ("lagrangian", "mlp", "rssm")
NOMINAL = "cartpole-swingup-random-100k-v2"
HORIZONS = (1, 5, 10, 20)


def _load(model: str, dataset: str) -> dict[str, Any] | None:
    path = EVAL_ROOT / model / dataset / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["metrics"]


def _probe(metrics: dict[str, Any], latent: str, target: str, kind: str) -> float | None:
    try:
        return float(metrics["state_probes"]["probes"][latent][target][kind]["r2_overall"])
    except KeyError:
        return None


def _f(value: float | None, digits: int = 5) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pose_latent(model: str) -> str:
    return "full" if model == "rssm" else "q"


def build_report() -> tuple[str, dict[str, Any]]:
    datasets = sorted(p.name for p in (EVAL_ROOT / "rssm").glob("*") if p.is_dir()) if (
        EVAL_ROOT / "rssm"
    ).exists() else []
    ood = [d for d in datasets if d != NOMINAL]
    nominal = {m: _load(m, NOMINAL) for m in MODELS}
    lines = ["# Gate experiment report", ""]

    lines += [
        "## Nominal (val split, 256 windows, warmup 5)",
        "",
        "| model | " + " | ".join(f"fg h{h}" for h in HORIZONS)
        + " | full h20 | pose probe MLP | pose probe linear | z→pose MLP | qvel probe MLP |",
        "|---|" + "---:|" * (len(HORIZONS) + 5),
    ]
    for model in MODELS:
        m = nominal[model]
        if m is None:
            lines.append(f"| {model} | " + " | ".join(["missing"] * (len(HORIZONS) + 5)) + " |")
            continue
        latent = _pose_latent(model)
        lines.append(
            f"| {model} | "
            + " | ".join(_f(m.get(f"foreground_open_loop_mse_h{h}")) for h in HORIZONS)
            + f" | {_f(m.get('open_loop_mse_h20'))}"
            + f" | {_f(_probe(m, latent, 'pose', 'mlp'), 3)}"
            + f" | {_f(_probe(m, latent, 'pose', 'linear'), 3)}"
            + f" | {_f(_probe(m, 'z', 'pose', 'mlp'), 3) if model != 'rssm' else '—'}"
            + f" | {_f(_probe(m, 'full', 'qvel', 'mlp'), 3)} |"
        )
    copy = next((m for m in nominal.values() if m is not None), None)
    if copy is not None:
        lines.append(
            "| copy last frame | "
            + " | ".join(_f(copy.get(f"foreground_copy_last_frame_mse_h{h}")) for h in HORIZONS)
            + f" | {_f(copy.get('copy_last_frame_mse_h20'))} | | | | |"
        )
    lines += [
        "",
        "Pose probe: MLP / linear R² from the model's mechanics coordinate `q` "
        "(RSSM: full feature) onto `[x, cos θ, sin θ]`, episode-level split.",
        "",
    ]

    ratios: dict[str, float | None] = {}
    if ood:
        lines += [
            "## OOD (unseen physics; train split of each OOD-v2 set)",
            "",
            "| dataset | " + " | ".join(f"{m} fg h20 (×nominal)" for m in MODELS) + " |",
            "|---|" + "---:|" * len(MODELS),
        ]
        per_model: dict[str, list[float]] = {m: [] for m in MODELS}
        for dataset in ood:
            cells = []
            for model in MODELS:
                m = _load(model, dataset)
                base = nominal[model]
                if m is None or base is None:
                    cells.append("missing")
                    continue
                value = m["foreground_open_loop_mse_h20"]
                ratio = value / base["foreground_open_loop_mse_h20"]
                per_model[model].append(ratio)
                cells.append(f"{value:.5f} (×{ratio:.2f})")
            lines.append(f"| {dataset.removeprefix('cartpole-ood-')} | " + " | ".join(cells) + " |")
        for model, values in per_model.items():
            ratios[model] = sum(values) / len(values) if values else None
        lines += [
            "| **mean degradation** | "
            + " | ".join(f"×{ratios[m]:.2f}" if ratios[m] is not None else "—" for m in MODELS)
            + " |",
            "",
        ]

    lag, mlp, rssm = nominal["lagrangian"], nominal["mlp"], nominal["rssm"]
    criteria: dict[str, bool | None] = {}
    if lag is not None:
        probe = _probe(lag, "q", "pose", "mlp")
        criteria["1. q carries the pose (MLP R² > 0.8)"] = (
            None if probe is None else probe > 0.8
        )
    if lag is not None and rssm is not None:
        criteria["2. nominal fg h20 <= RSSM"] = (
            lag["foreground_open_loop_mse_h20"] <= rssm["foreground_open_loop_mse_h20"]
        )
    if ratios.get("lagrangian") is not None and ratios.get("rssm") is not None:
        criteria["3. OOD degradation < RSSM"] = ratios["lagrangian"] < ratios["rssm"]
    if lag is not None and mlp is not None:
        criteria["vs MLP: nominal fg h20 <= MLP arm"] = (
            lag["foreground_open_loop_mse_h20"] <= mlp["foreground_open_loop_mse_h20"]
        )
    if ratios.get("lagrangian") is not None and ratios.get("mlp") is not None:
        criteria["vs MLP: OOD degradation < MLP arm"] = ratios["lagrangian"] < ratios["mlp"]

    lines += ["## Gate", ""]
    for name, passed in criteria.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL' if passed is not None else 'n/a'}")
    complete = len(criteria) == 5 and all(v is not None for v in criteria.values())
    verdict = (
        "INCOMPLETE (missing evaluations)"
        if not complete
        else "PASS — continue to the MPC phase"
        if all(criteria.values())
        else "FAIL — stop criterion applies (see docs/current_status.md)"
    )
    lines += ["", f"**Verdict: {verdict}**", ""]
    return "\n".join(lines), {"criteria": criteria, "verdict": verdict, "ood_ratios": ratios}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report, summary = build_report()
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        args.output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()
