"""Aggregate result records across seeds into tables (no hand-typed numbers).

Runs are grouped into *conditions*: identical configs once per-seed keys
(``seed``, and outputs that some oracles store in their config, such as
``recipes``) are removed. Per (condition, seed) only the latest run counts;
earlier ones are listed as superseded. Metric patterns may contain
``{shift}``, expanded over each run's ``config["shifts"]``, and a ratio
``a / b`` of two metric names, computed per seed before averaging.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

PER_SEED_KEYS = ("seed", "recipes")


@dataclass
class Condition:
    config: dict[str, Any]
    runs: dict[int | None, Path] = field(default_factory=dict)  # seed -> latest result file
    payloads: dict[int | None, dict[str, Any]] = field(default_factory=dict)
    superseded: list[Path] = field(default_factory=list)


def _canonical(config: dict[str, Any], drop: Sequence[str]) -> str:
    return json.dumps({k: v for k, v in config.items() if k not in drop}, sort_keys=True, default=str)


def group_runs(
    paths: Iterable[Path], drop: Sequence[str] = PER_SEED_KEYS, seeds: Iterable[Any] | None = None,
) -> list[Condition]:
    """Group result files by config minus ``drop``; ``seeds`` keeps only runs with those seeds
    (e.g. evaluation seeds, leaving out tuning runs that share a config)."""

    keep = None if seeds is None else set(seeds)
    conditions: dict[str, Condition] = {}
    for path in sorted(paths):  # file names are timestamps, so later files win
        if path.name == "latest.json":
            continue
        payload = json.loads(path.read_text())
        config = payload.get("config", {})
        if keep is not None and config.get("seed") not in keep:
            continue
        key = _canonical(config, drop)
        condition = conditions.setdefault(key, Condition(config={k: v for k, v in config.items() if k not in drop}))
        seed = config.get("seed")
        if seed in condition.runs:
            condition.superseded.append(condition.runs[seed])
        condition.runs[seed] = path
        condition.payloads[seed] = payload
    return list(conditions.values())


def filter_conditions(conditions: list[Condition], where: dict[str, Any]) -> list[Condition]:
    def matches(config: dict[str, Any]) -> bool:
        return all(_loose_equal(config.get(k), v) for k, v in where.items())

    return [c for c in conditions if matches(c.config)]


def _loose_equal(value: Any, wanted: Any) -> bool:
    if isinstance(wanted, str) and not isinstance(value, str):
        return json.dumps(value, default=str) == wanted or str(value) == wanted
    return value == wanted


def parse_seeds(text: str | None) -> list[int] | None:
    """``"0-4"`` or ``"0,1,5"`` (or a mix) to a list of ints; ``None`` keeps all seeds."""

    if not text:
        return None
    seeds: list[int] = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        seeds.extend(range(int(low), int(high) + 1) if high else [int(low)])
    return seeds


def parse_where(items: Sequence[str]) -> dict[str, Any]:
    """``key=value`` with JSON values (``null``, numbers, lists), falling back to strings."""

    parsed: dict[str, Any] = {}
    for item in items:
        key, _, raw = item.partition("=")
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    return parsed


def _metric_value(metrics: dict[str, Any], expr: str) -> float:
    if " / " in expr:
        num, den = (part.strip() for part in expr.split(" / ", 1))
        a, b = _metric_value(metrics, num), _metric_value(metrics, den)
        return a / b if b not in (0.0,) and math.isfinite(a) and math.isfinite(b) else math.nan
    value = metrics.get(expr)
    return float(value) if isinstance(value, (int, float)) and value is not None else math.nan


def expand(pattern: str, config: dict[str, Any]) -> list[tuple[str, str]]:
    """``(label, expression)`` pairs for one pattern; ``{shift}`` expands over the run's shifts."""

    if "{shift}" not in pattern:
        return [(pattern, pattern)]
    return [(pattern.replace("{shift}", s), pattern.replace("{shift}", s)) for s in config.get("shifts", [])]


def summarize(condition: Condition, patterns: Sequence[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        if pattern.startswith("re:"):
            regex = re.compile(pattern[3:])
            names = sorted({m for p in condition.payloads.values() for m in p.get("metrics", {}) if regex.search(m)})
            pairs = [(n, n) for n in names]
        else:
            pairs = expand(pattern, condition.config)
        for label, expr in pairs:
            values = {seed: _metric_value(p.get("metrics", {}), expr) for seed, p in condition.payloads.items()}
            finite = [v for v in values.values() if math.isfinite(v)]
            n = len(finite)
            mean = sum(finite) / n if n else math.nan
            std = math.sqrt(sum((v - mean) ** 2 for v in finite) / (n - 1)) if n > 1 else math.nan
            out[label] = {
                "mean": mean, "std": std, "n": n,
                "values": {str(seed): v for seed, v in sorted(values.items(), key=lambda kv: str(kv[0]))},
            }
    return out


def varying_keys(conditions: Sequence[Condition]) -> list[str]:
    keys = sorted({k for c in conditions for k in c.config})
    return [k for k in keys if len({json.dumps(c.config.get(k), default=str) for c in conditions}) > 1]


def _short(value: Any) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= 60 else text[:57] + "..."


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.3g}" if abs(value) >= 1e-3 else f"{value:.2e}"


def markdown(conditions: Sequence[Condition], patterns: Sequence[str]) -> str:
    """One table per condition: metric, mean ± sample std (n), per-seed values."""

    keys = varying_keys(conditions)
    blocks = []
    for index, condition in enumerate(conditions):
        label = ", ".join(f"`{k}`={_short(condition.config.get(k))}" for k in keys) or "single condition"
        runs = ", ".join(f"`{p.parent.name}/{p.stem}`" for _, p in sorted(condition.runs.items(), key=lambda kv: str(kv[0])))
        lines = [f"**Condition {index + 1}:** {label}", "", f"Runs (seed order): {runs}"]
        if condition.superseded:
            lines.append("Superseded: " + ", ".join(f"`{p.parent.name}/{p.stem}`" for p in condition.superseded))
        lines += ["", "| metric | mean ± std (n) | per seed |", "|---|---|---|"]
        for name, stats in summarize(condition, patterns).items():
            per_seed = " / ".join(_fmt(v) for v in stats["values"].values())
            lines.append(f"| `{name}` | {_fmt(stats['mean'])} ± {_fmt(stats['std'])} ({stats['n']}) | {per_seed} |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n\n_Generated by `scripts/research/report.py`; std is the sample std over seeds._"


def summary_json(conditions: Sequence[Condition], patterns: Sequence[str]) -> dict[str, Any]:
    return {
        "patterns": list(patterns),
        "conditions": [
            {
                "config": c.config,
                "runs": {str(seed): f"{p.parent.name}/{p.stem}" for seed, p in c.runs.items()},
                "superseded": [f"{p.parent.name}/{p.stem}" for p in c.superseded],
                "metrics": summarize(c, patterns),
            }
            for c in conditions
        ],
    }


def report_patterns(record_text: str) -> list[str]:
    """Backticked metric patterns on the ``Report:`` line of a record's *Primary metric* section."""

    from research.records import sections

    body = sections(record_text).get("Primary metric", "")
    for line in body.splitlines():
        if line.strip().lower().startswith("report:"):
            return re.findall(r"`([^`]+)`", line)
    return []


def resolve_paths(entries: Iterable[dict[str, Any]], repo: Path) -> list[Path]:
    return [repo / e["path"] for e in entries if (repo / e["path"]).exists()]
