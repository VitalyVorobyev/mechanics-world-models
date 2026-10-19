"""The committed run ledger: one JSON line per recorded result.

Result files live in the gitignored ``runs/``; the ledger (``results/ledger.jsonl``)
is committed, so the history of what was run, from which commit, and with
which outcome survives even where the result files do not. Each line carries
the sha256 of its result file, so an edited file is detectable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research.provenance import REPO_ROOT

LEDGER_PATH = REPO_ROOT / "results" / "ledger.jsonl"


def run_id(result_path: Path) -> str:
    """``<oracle name>/<file stem>``, e.g. ``context_oracle/20261018-182854``."""

    return f"{result_path.parent.name}/{result_path.stem}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def entry_for(result_path: Path, payload: dict[str, Any], *, legacy: bool = False) -> dict[str, Any]:
    provenance = payload.get("provenance") or {}
    config = payload.get("config") or {}
    verdict = payload.get("verdict") or {}
    return {
        "run_id": run_id(result_path),
        "path": _relative(result_path),
        "sha256": file_sha256(result_path),
        "name": payload.get("name"),
        "experiment": payload.get("experiment"),
        "hypotheses": payload.get("hypotheses", []),
        "timestamp": payload.get("timestamp"),
        "commit": provenance.get("commit") or payload.get("git_commit"),
        "dirty": provenance.get("dirty") if provenance else None,
        "seed": config.get("seed"),
        "passed": verdict.get("passed"),
        "legacy": legacy,
    }


def load(path: Path = LEDGER_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(entry: dict[str, Any], path: Path = LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def by_id(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["run_id"]: entry for entry in entries}
