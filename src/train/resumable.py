"""Helpers that make long training runs interruptible, resumable and observable.

* ``EpochResumableSampler`` — a per-epoch seeded permutation that can start
  part-way through an epoch, so a run resumed from a mid-epoch checkpoint
  sees exactly the batches it had not processed yet.
* ``GracefulStop`` — turns the first SIGINT/SIGTERM into a "stop at the next
  step boundary" flag so the trainer can write a checkpoint and exit
  cleanly; a second signal falls through to the default handler.
* ``atomic_torch_save`` — write-then-rename so an interrupted save never
  leaves a truncated ``latest.pt``.
* ``truncate_history`` — drop JSONL rows logged after the checkpoint being
  resumed, so a hard crash does not leave duplicated steps in the history.
* ``ProgressTracker`` — rewrites a small ``progress.json`` (step, rate, ETA,
  latest metrics) that status tools can poll without parsing the history.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Sampler


class EpochResumableSampler(Sampler[int]):
    """Shuffle with a generator seeded by ``(seed, epoch)``; optionally skip a prefix."""

    def __init__(self, num_samples: int, seed: int) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch: int, skip_samples: int = 0) -> None:
        """Select the epoch's permutation and how many leading samples to skip."""

        if not 0 <= skip_samples <= self.num_samples:
            raise ValueError("skip_samples must lie in [0, num_samples]")
        self.epoch = epoch
        self.start = skip_samples

    def _permutation(self) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(self.seed * 1_000_003 + self.epoch)
        return torch.randperm(self.num_samples, generator=generator)

    def __iter__(self) -> Iterator[int]:
        return iter(self._permutation()[self.start :].tolist())

    def __len__(self) -> int:
        return self.num_samples - self.start


class GracefulStop:
    """Context manager that records the first SIGINT/SIGTERM instead of raising."""

    def __init__(self) -> None:
        self.requested = False
        self._previous: dict[int, Any] = {}

    def _handle(self, signum: int, frame: Any) -> None:
        if self.requested:
            # Second signal: restore defaults and re-raise the usual behaviour.
            self._restore()
            signal.raise_signal(signum)
            return
        self.requested = True

    def _restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def __enter__(self) -> "GracefulStop":
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.signal(signum, self._handle)
        return self

    def __exit__(self, *exc: object) -> None:
        self._restore()


def atomic_torch_save(payload: Any, path: Path) -> None:
    """``torch.save`` to a sibling temp file, then atomically rename over ``path``."""

    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def truncate_history(history_path: Path, max_global_step: int) -> int:
    """Keep only JSONL rows with ``global_step <= max_global_step``; return rows dropped."""

    if not history_path.exists():
        return 0
    kept: list[str] = []
    dropped = 0
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            step = int(json.loads(line).get("global_step", 0))
        except (ValueError, json.JSONDecodeError):
            dropped += 1  # a line cut off by a hard kill
            continue
        if step <= max_global_step:
            kept.append(line)
        else:
            dropped += 1
    if dropped:
        tmp_path = history_path.with_name(history_path.name + ".tmp")
        tmp_path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
        os.replace(tmp_path, history_path)
    return dropped


class ProgressTracker:
    """Maintain ``progress.json`` with run position, throughput and ETA."""

    def __init__(self, path: Path, *, total_steps: int, start_step: int, total_epochs: int) -> None:
        self.path = path
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self._start_step = start_step
        self._start_time = time.monotonic()
        self._latest: dict[str, float] = {}
        self._probe: dict[str, float] = {}

    def update(
        self,
        *,
        epoch: int,
        global_step: int,
        status: str = "running",
        metrics: dict[str, float] | None = None,
        probe: dict[str, float] | None = None,
    ) -> None:
        if metrics:
            self._latest = {k: float(v) for k, v in metrics.items()}
        if probe:
            self._probe = {k: float(v) for k, v in probe.items()}
        elapsed = time.monotonic() - self._start_time
        done = global_step - self._start_step
        rate = done / elapsed if elapsed > 0 and done > 0 else None
        remaining = max(self.total_steps - global_step, 0)
        payload = {
            "status": status,
            "pid": os.getpid(),
            "updated_unix": time.time(),
            "epoch": epoch,
            "total_epochs": self.total_epochs,
            "global_step": global_step,
            "total_steps": self.total_steps,
            "fraction_done": global_step / self.total_steps if self.total_steps else None,
            "steps_per_second_this_session": rate,
            "eta_seconds": remaining / rate if rate else None,
            "latest_train": self._latest,
            "latest_val_open_loop": self._probe,
        }
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, self.path)
