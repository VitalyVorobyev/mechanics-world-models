"""Where a result came from: commit, tree state, command line, versions, data and code fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

# Source trees whose edits can change an oracle's numbers (cache keys and records include their hash).
ORACLE_SOURCE_DIRS = (REPO_ROOT / "src" / "oracles", REPO_ROOT / "src" / "models" / "mechanics")
_VERSIONED_PACKAGES = ("torch", "numpy", "mujoco", "dm-control")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_commit() -> str | None:
    return _git("rev-parse", "HEAD")


def git_dirty_paths() -> list[str]:
    """Tracked files with uncommitted changes, plus untracked files under ``src/`` and ``scripts/``."""

    status = _git("status", "--porcelain", "--untracked-files=all", "--", "src", "scripts", "pyproject.toml")
    if not status:
        return []
    return [line[3:] for line in status.splitlines() if line.strip()]


def source_fingerprint(dirs: Iterable[Path] = ORACLE_SOURCE_DIRS) -> str:
    """sha1 over the contents of every ``*.py`` file under ``dirs`` (paths relative to the repo)."""

    digest = hashlib.sha1()
    for root in dirs:
        for path in sorted(Path(root).rglob("*.py")):
            digest.update(str(path.relative_to(REPO_ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def directory_fingerprint(path: Path) -> str:
    """Cheap dataset fingerprint: sha1 over sorted (relative name, size) of the files in ``path``.

    It detects a recollected or edited dataset, not bit flips inside a file of the same size.
    """

    digest = hashlib.sha1()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(f"{item.relative_to(path)}:{item.stat().st_size}\n".encode())
    return digest.hexdigest()[:12]


def data_fingerprints(config: dict[str, Any]) -> dict[str, str]:
    """Fingerprint every config value (also inside lists, and ``name=dir`` items) that names a directory."""

    found: dict[str, str] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, (str, Path)):
            candidate = Path(str(value).partition("=")[2] or str(value))
            if len(str(candidate)) < 512 and candidate.is_dir():
                found[str(candidate)] = directory_fingerprint(candidate)

    visit(config)
    return found


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _VERSIONED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@dataclass
class RunContext:
    """Captured when a run starts; attached to its result record and ledger line."""

    experiment: str | None
    commit: str | None
    dirty_paths: list[str]
    argv: list[str]
    started: str
    python: str = field(default_factory=lambda: sys.version.split()[0])

    @property
    def dirty(self) -> bool:
        return bool(self.dirty_paths)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dirty"] = self.dirty
        return payload


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """``--experiment E-NNN`` (the registered record this run tests) and ``--allow-dirty``."""

    parser.add_argument(
        "--experiment", default=None,
        help="ID of the registered experiment record (docs/experiments/E-NNN-*.md) this run belongs to",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="run with uncommitted changes under src/ or scripts/ (the result is recorded as ungradeable)",
    )


def begin_run(args: argparse.Namespace | None = None, *, argv: list[str] | None = None) -> RunContext:
    """Capture provenance at the start of a run; refuse a dirty tree unless ``--allow-dirty``.

    A result from uncommitted code cannot be reproduced from its recorded
    commit, so it is refused up front instead of discovered later.
    """

    experiment = getattr(args, "experiment", None) if args is not None else None
    allow_dirty = bool(getattr(args, "allow_dirty", False)) if args is not None else True
    context = RunContext(
        experiment=experiment,
        commit=git_commit(),
        dirty_paths=git_dirty_paths(),
        argv=list(sys.argv if argv is None else argv),
        started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    if context.dirty and not allow_dirty:
        listed = "\n  ".join(context.dirty_paths[:20])
        raise SystemExit(
            "refusing to run with uncommitted code (commit first, or pass --allow-dirty to record an "
            f"ungradeable result):\n  {listed}"
        )
    if experiment is not None:
        from research.records import find_record  # local import: records imports nothing from here

        if find_record(experiment) is None:
            raise SystemExit(f"no experiment record for {experiment!r} under docs/experiments/")
    return context
