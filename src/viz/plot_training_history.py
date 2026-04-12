"""Plot RSSM training-history JSONL files."""

from __future__ import annotations

import argparse
import json
import os
from math import isfinite
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Sequence

_CACHE_ROOT = Path(gettempdir()) / "mechanics-world-models-cache"
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_METRICS = (
    "total_loss",
    "reconstruction_loss",
    "weighted_reconstruction_loss",
    "foreground_reconstruction_loss",
    "weighted_foreground_reconstruction_loss",
    "transition_reconstruction_loss",
    "weighted_transition_reconstruction_loss",
    "foreground_transition_reconstruction_loss",
    "weighted_foreground_transition_reconstruction_loss",
    "dynamic_reconstruction_loss",
    "weighted_dynamic_reconstruction_loss",
    "kl_loss",
    "weighted_kl_loss",
    "latent_consistency_loss",
    "weighted_latent_consistency_loss",
)
DEFAULT_SPLITS = ("train", "train_epoch", "val")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command line parser for training-history plotting."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--log-y", action="store_true")
    return parser


def load_history(history_path: Path) -> list[dict[str, Any]]:
    """Load line-delimited JSON metric records from a training run."""

    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")

    records: list[dict[str, Any]] = []
    with history_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {history_path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"History line {line_number} must contain a JSON object")
            records.append(record)

    if not records:
        raise ValueError(f"Training history is empty: {history_path}")
    return records


def plot_training_history(
    history_path: Path,
    output_path: Path,
    metrics: Sequence[str] = DEFAULT_METRICS,
    splits: Sequence[str] = DEFAULT_SPLITS,
    log_y: bool = False,
) -> Path:
    """Plot selected loss metrics against optimizer ``global_step``."""

    records = load_history(history_path)
    selected_metrics = [metric for metric in metrics if metric_has_values(records, metric)]
    if not selected_metrics:
        raise ValueError(f"No requested metrics found in {history_path}")

    fig, axes = plt.subplots(
        nrows=len(selected_metrics),
        ncols=1,
        figsize=(8.0, max(2.4 * len(selected_metrics), 3.0)),
        sharex=True,
        squeeze=False,
    )
    line_count = 0
    for axis, metric in zip(axes[:, 0], selected_metrics):
        axis_line_count = 0
        for split in splits:
            points = metric_points(records, split=split, metric=metric)
            if not points:
                continue
            steps, values = zip(*points)
            axis.plot(steps, values, marker=".", linewidth=1.2, markersize=4, label=split)
            line_count += 1
            axis_line_count += 1
        axis.set_ylabel(metric)
        if log_y:
            axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
        if axis_line_count > 0:
            axis.legend(loc="best")

    if line_count == 0:
        raise ValueError(f"No requested split/metric pairs found in {history_path}")

    axes[-1, 0].set_xlabel("global_step")
    fig.suptitle(history_path.name)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def metric_has_values(records: Sequence[dict[str, Any]], metric: str) -> bool:
    """Return whether any record contains a numeric value for ``metric``."""

    return any(is_number(record.get(metric)) for record in records)


def metric_points(
    records: Sequence[dict[str, Any]],
    split: str,
    metric: str,
) -> list[tuple[int, float]]:
    """Return ``(global_step, value)`` points for one split and metric."""

    points: list[tuple[int, float]] = []
    for record in records:
        if record.get("split") != split:
            continue
        step = record.get("global_step")
        value = record.get(metric)
        if not is_number(step) or not is_number(value):
            continue
        points.append((int(step), float(value)))
    return sorted(points, key=lambda item: item[0])


def is_number(value: object) -> bool:
    """Return whether ``value`` is a finite int or float, excluding booleans."""

    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value))


def default_output_path(history_path: Path) -> Path:
    """Return the default plot path next to a history file."""

    return history_path.with_name("loss_history.png")


def main() -> None:
    """Run the training-history plotting CLI."""

    args = build_arg_parser().parse_args()
    output_path = args.output_path or default_output_path(args.history_path)
    path = plot_training_history(
        history_path=args.history_path,
        output_path=output_path,
        metrics=args.metrics,
        splits=args.splits,
        log_y=args.log_y,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
