"""Per-step wall-clock benchmark for ``MechanicsWorldModel`` training.

The mechanics model's dynamics loop is the dominant per-step cost on MPS
(see ``docs/models/mechanics_world_model.md``), so we want a reproducible
way to track step latency across refactors. This script:

* builds a model with the same defaults as ``train-mechanics``,
* runs ``warmup_steps`` of forward+backward+optimizer to let the backend
  JIT and stabilize allocator behaviour,
* times ``steps`` more steps end-to-end with proper accelerator sync
  around each step, and
* reports min / median / mean / max step time, plus a sub-phase split
  for the median step (forward, backward, grad-norm + optimizer).

Run for example:

    .venv/bin/python scripts/bench_mechanics_step.py --device mps --steps 20

Synthetic random data is used — no dataset on disk is required.
"""

from __future__ import annotations

import argparse
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from models.mechanics import (
    MechanicsWorldModel,
    compute_mechanics_world_model_loss,
)


@dataclass
class StepTimings:
    """Median-step sub-phase breakdown in milliseconds."""

    forward_ms: float
    backward_ms: float
    grad_opt_ms: float


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def _timed(device: torch.device, accumulator: list[float]):
    _sync(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync(device)
        accumulator.append((time.perf_counter() - start) * 1000.0)


def build_model(args: argparse.Namespace, device: torch.device) -> MechanicsWorldModel:
    """Instantiate the mechanics model with ``train-mechanics`` defaults."""

    return MechanicsWorldModel(
        action_dim=args.action_dim,
        q_dim=args.q_dim,
        nuisance_dim=args.nuisance_dim,
        embedding_size=args.embedding_size,
        encoder_head_hidden_size=args.encoder_head_hidden_size,
        mass_matrix_hidden_size=args.mass_matrix_hidden_size,
        potential_hidden_size=args.potential_hidden_size,
        dissipation_hidden_size=args.dissipation_hidden_size,
        actuation_hidden_size=args.actuation_hidden_size,
        dynamics_hidden_layers=args.dynamics_hidden_layers,
        image_size=args.image_size,
        dt=args.dt,
        with_dissipation=not args.no_dissipation,
    ).to(device)


def benchmark(
    model: MechanicsWorldModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[float], StepTimings]:
    """Run warmup + timed steps and return per-step latencies (ms) + sub-phase split."""

    torch.manual_seed(args.seed)
    obs = torch.rand(
        args.batch_size, args.sequence_length, 3, args.image_size, args.image_size,
        device=device,
    )
    nxt = torch.rand_like(obs)
    actions = torch.randn(
        args.batch_size, args.sequence_length, args.action_dim, device=device,
    )

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            obs, actions,
            imagination_context_steps=args.imagination_context_steps,
            imagination_horizon=args.imagination_horizon,
        )
        losses = compute_mechanics_world_model_loss(
            outputs=outputs,
            observations=obs,
            next_observations=nxt,
            foreground_imagination_reconstruction_weight=2.0,
            foreground_reconstruction_weight=1.0,
            smoothness_weight=1.0,
            dt=model.dt,
        )
        losses["total_loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    for _ in range(args.warmup_steps):
        one_step()
    _sync(device)

    step_times: list[float] = []
    for _ in range(args.steps):
        _sync(device)
        start = time.perf_counter()
        one_step()
        _sync(device)
        step_times.append((time.perf_counter() - start) * 1000.0)

    # Sub-phase split for one extra step (so the timings printed below add up
    # roughly to the median step). Done outside the main loop so the per-step
    # timings stay free of the extra sync overhead.
    forward_times: list[float] = []
    backward_times: list[float] = []
    grad_opt_times: list[float] = []
    for _ in range(3):
        with _timed(device, forward_times):
            outputs = model(
                obs, actions,
                imagination_context_steps=args.imagination_context_steps,
                imagination_horizon=args.imagination_horizon,
            )
            losses = compute_mechanics_world_model_loss(
                outputs=outputs,
                observations=obs,
                next_observations=nxt,
                foreground_imagination_reconstruction_weight=2.0,
                foreground_reconstruction_weight=1.0,
                smoothness_weight=1.0,
                dt=model.dt,
            )
        with _timed(device, backward_times):
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
        with _timed(device, grad_opt_times):
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

    timings = StepTimings(
        forward_ms=statistics.median(forward_times),
        backward_ms=statistics.median(backward_times),
        grad_opt_ms=statistics.median(grad_opt_times),
    )
    return step_times, timings


def report(step_times: list[float], timings: StepTimings, args: argparse.Namespace) -> None:
    """Print a compact text report. No JSON yet — keep it greppable."""

    n = len(step_times)
    print(f"device={args.device} steps={n} batch={args.batch_size} "
          f"T={args.sequence_length} q_dim={args.q_dim} "
          f"nuisance_dim={args.nuisance_dim} K_imag={args.imagination_horizon}")
    print(
        f"step_ms  min={min(step_times):.2f} "
        f"median={statistics.median(step_times):.2f} "
        f"mean={statistics.mean(step_times):.2f} "
        f"max={max(step_times):.2f}"
    )
    if n >= 3:
        print(
            f"phase_ms forward={timings.forward_ms:.2f} "
            f"backward={timings.backward_ms:.2f} "
            f"grad+opt={timings.grad_opt_ms:.2f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--imagination-context-steps", type=int, default=4)
    parser.add_argument("--imagination-horizon", type=int, default=12)
    parser.add_argument("--action-dim", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--q-dim", type=int, default=2)
    parser.add_argument("--nuisance-dim", type=int, default=32)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--encoder-head-hidden-size", type=int, default=128)
    parser.add_argument("--mass-matrix-hidden-size", type=int, default=64)
    parser.add_argument("--potential-hidden-size", type=int, default=64)
    parser.add_argument("--dissipation-hidden-size", type=int, default=64)
    parser.add_argument("--actuation-hidden-size", type=int, default=64)
    parser.add_argument("--dynamics-hidden-layers", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--no-dissipation", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = _resolve_device(args.device)
    model = build_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    step_times, timings = benchmark(model, optimizer, args, device)
    report(step_times, timings, args)


if __name__ == "__main__":
    main()
