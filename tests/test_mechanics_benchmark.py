"""Smoke test for ``scripts/bench_mechanics_step.py``.

Pure runnability check. We deliberately do not assert wall-clock numbers
because they depend on the host machine and the test would be flaky in CI.
What we *do* care about is that the harness builds the model the same way
``train-mechanics`` does and survives a real forward+backward+optimizer
loop end-to-end on CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add scripts/ to sys.path so we can import the benchmark as a module without
# spawning a subprocess (faster, and surfaces tracebacks directly).
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import bench_mechanics_step as bench  # noqa: E402


def _tiny_args() -> argparse.Namespace:
    """Build a tiny-but-valid args namespace so the smoke test stays fast."""

    parser = bench.build_arg_parser()
    return parser.parse_args(
        [
            "--device", "cpu",
            "--steps", "2",
            "--warmup-steps", "1",
            "--batch-size", "2",
            "--sequence-length", "8",
            "--imagination-context-steps", "2",
            "--imagination-horizon", "3",
            "--nuisance-dim", "4",
            "--embedding-size", "32",
            "--encoder-head-hidden-size", "16",
            "--mass-matrix-hidden-size", "16",
            "--potential-hidden-size", "16",
            "--dissipation-hidden-size", "16",
            "--actuation-hidden-size", "16",
            "--dynamics-hidden-layers", "1",
        ],
    )


def test_benchmark_runs_end_to_end() -> None:
    args = _tiny_args()
    device = bench._resolve_device(args.device)
    model = bench.build_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    step_times, timings = bench.benchmark(model, optimizer, args, device)

    assert len(step_times) == args.steps
    assert all(t > 0 for t in step_times)
    assert timings.forward_ms > 0
    assert timings.backward_ms > 0
    assert timings.grad_opt_ms > 0
