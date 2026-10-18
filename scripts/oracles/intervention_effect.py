"""Rung 0 CLI: how much does each physics perturbation move cartpole trajectories?

Usage:
  .venv/bin/python scripts/oracles/intervention_effect.py \
      --shift pole_mass_x2:body_mass_2=2.0 --shift cart_mass_x05:body_mass_1=0.5

Each --shift is NAME:KEY=SCALE[,KEY=SCALE...] with keys accepted by
--physics-scale (pole_length, body_mass_<i>, dof_damping_<i>, geom_size_<i>_<j>).
Writes runs/oracles/intervention_effect/<timestamp>.json.
"""

from __future__ import annotations

import argparse

from oracles.intervention import run_intervention_effect

DEFAULT_SHIFTS = [
    "pole_mass_x07:body_mass_2=0.7",
    "pole_mass_x2:body_mass_2=2.0",
    "cart_mass_x05:body_mass_1=0.5",
    "cart_mass_x2:body_mass_1=2.0",
    "pole_length_x07:pole_length=0.7",
    "pole_length_x13:pole_length=1.3",
    "pole_length_x15:pole_length=1.5",
    "hinge_damping_x100:dof_damping_1=100",
    "slider_damping_x100:dof_damping_0=100",
]


def _parse_shift(text: str) -> tuple[str, dict[str, float]]:
    name, _, spec = text.partition(":")
    scales = {}
    for item in spec.split(","):
        key, _, value = item.partition("=")
        scales[key] = float(value)
    return name, scales


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shift", action="append", default=None)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--min-effect-h20", type=float, default=0.01)
    args = parser.parse_args()
    shifts = dict(_parse_shift(s) for s in (args.shift or DEFAULT_SHIFTS))
    result = run_intervention_effect(
        shifts, seeds=range(args.seeds), min_effect_h20=args.min_effect_h20,
    )
    path = result.write()
    for name in shifts:
        devs = "  ".join(
            f"h{h} {result.metrics[f'{name}.mean_abs_dev_h{h}']:.4f}" for h in (10, 20, 50)
        )
        print(f"{name:22s} {devs}")
    print(result.summary())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
