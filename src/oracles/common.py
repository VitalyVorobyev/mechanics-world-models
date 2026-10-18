"""Shared helpers for oracle diagnostics: result records and ground-truth data."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from data.sequence_dataset import index_episode_files, split_episode_files
from data.trajectory import load_trajectory_npz

DEFAULT_RESULTS_ROOT = Path("runs/oracles")


@dataclass
class Criterion:
    """One pass/fail check: ``metric`` compared to ``threshold``."""

    metric: str
    op: str  # "<=", ">=", "<", ">"
    threshold: float
    description: str = ""

    def check(self, metrics: dict[str, Any]) -> bool | None:
        value = metrics.get(self.metric)
        if value is None or not np.isfinite(value):
            return None
        return bool({
            "<=": value <= self.threshold,
            ">=": value >= self.threshold,
            "<": value < self.threshold,
            ">": value > self.threshold,
        }[self.op])


@dataclass
class OracleResult:
    """A self-describing oracle record (written as JSON)."""

    name: str
    hypotheses: list[str]
    question: str
    config: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    criteria: list[Criterion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def verdict(self) -> dict[str, Any]:
        checks = {f"{c.metric} {c.op} {c.threshold}": c.check(self.metrics) for c in self.criteria}
        values = list(checks.values())
        passed = None if any(v is None for v in values) else all(values)
        return {"checks": checks, "passed": passed}

    def write(self, root: Path = DEFAULT_RESULTS_ROOT) -> Path:
        out_dir = root / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        payload = {
            "name": self.name,
            "hypotheses": self.hypotheses,
            "question": self.question,
            "timestamp": stamp,
            "git_commit": _git_commit(),
            "config": self.config,
            "metrics": self.metrics,
            "criteria": [vars(c) for c in self.criteria],
            "verdict": self.verdict(),
            "notes": self.notes,
        }
        path = out_dir / f"{stamp}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
        (out_dir / "latest.json").write_text(path.read_text())
        return path

    def summary(self) -> str:
        verdict = self.verdict()
        lines = [f"[{self.name}] {self.question}"]
        for check, ok in verdict["checks"].items():
            lines.append(f"  {'PASS' if ok else 'FAIL' if ok is not None else 'n/a '}  {check}")
        lines.append(f"  verdict: {verdict['passed']}")
        return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass
class StateEpisodes:
    """Ground-truth trajectories: ``qpos``/``qvel`` ``[E, N+1, 2]``, ``actions`` ``[E, N, A]``."""

    qpos: np.ndarray
    qvel: np.ndarray
    actions: np.ndarray
    frame_dt: float


def load_state_episodes(
    dataset_dir: Path,
    *,
    split: str | None = None,
    val_fraction: float = 0.1,
    seed: int = 0,
    max_episodes: int | None = None,
    episode_ids: Sequence[int] | None = None,
) -> StateEpisodes:
    """Load ``qpos``/``qvel``/``actions`` for (a split of) a dataset, no images."""

    files = index_episode_files(dataset_dir)
    if split is not None:
        files = split_episode_files(files, split, val_fraction=val_fraction, seed=seed)
    if episode_ids is not None:
        files = [files[i] for i in episode_ids]
    if max_episodes is not None:
        files = files[:max_episodes]
    qpos, qvel, actions, frame_dt = [], [], [], None
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if "qpos" not in data.files:
                raise ValueError(f"{path} has no physics state; recollect with a newer collector")
            qpos.append(np.asarray(data["qpos"], dtype=np.float64))
            qvel.append(np.asarray(data["qvel"], dtype=np.float64))
            actions.append(np.asarray(data["actions"], dtype=np.float64).reshape(len(data["actions"]), -1))
            if frame_dt is None:
                frame_dt = float(data["frame_dt"]) if "frame_dt" in data.files else 0.01 * int(data["action_repeat"])
    lengths = {q.shape[0] for q in qpos}
    if len(lengths) != 1:
        min_len = min(lengths)
        qpos = [q[:min_len] for q in qpos]
        qvel = [v[:min_len] for v in qvel]
        actions = [a[: min_len - 1] for a in actions]
    return StateEpisodes(
        qpos=np.stack(qpos), qvel=np.stack(qvel), actions=np.stack(actions), frame_dt=float(frame_dt),
    )


def load_frames_with_state(
    dataset_dir: Path,
    *,
    split: str,
    val_fraction: float = 0.1,
    seed: int = 0,
    max_episodes: int | None = None,
    frame_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(images uint8 [N, H, W, 3], qpos [N, 2], episode_index [N])`` for a split."""

    files = index_episode_files(dataset_dir)
    files = split_episode_files(files, split, val_fraction=val_fraction, seed=seed)
    if max_episodes is not None:
        files = files[:max_episodes]
    images, qpos, episode = [], [], []
    for index, path in enumerate(files):
        trajectory = load_trajectory_npz(path)
        if trajectory.qpos is None:
            raise ValueError(f"{path} has no physics state")
        images.append(trajectory.images[::frame_stride])
        qpos.append(trajectory.qpos[::frame_stride])
        episode.append(np.full(len(trajectory.qpos[::frame_stride]), index))
    return np.concatenate(images), np.concatenate(qpos), np.concatenate(episode)


def pose_features(qpos: np.ndarray) -> np.ndarray:
    """``[..., 2]`` (x, theta) -> ``[..., 3]`` (x, cos theta, sin theta)."""

    return np.stack([qpos[..., 0], np.cos(qpos[..., 1]), np.sin(qpos[..., 1])], axis=-1)


def load_shift_splits(dataset_dir: Path, seed: int = 0) -> tuple[StateEpisodes, StateEpisodes, StateEpisodes]:
    """Split a shifted dataset by episode: half adaptation, a quarter selection, a quarter test."""

    held_out = load_state_episodes(dataset_dir, split="val", val_fraction=0.5, seed=seed)
    half = held_out.qpos.shape[0] // 2

    def part(index: slice) -> StateEpisodes:
        return StateEpisodes(
            qpos=held_out.qpos[index], qvel=held_out.qvel[index], actions=held_out.actions[index],
            frame_dt=held_out.frame_dt,
        )

    train = load_state_episodes(dataset_dir, split="train", val_fraction=0.5, seed=seed)
    return train, part(slice(0, half)), part(slice(half, None))
