#!/bin/bash
# Closed-loop MPC evaluation grid for the H1 (physical OOD) and H2 (visual
# OOD) hypotheses from docs/research_spec.md. Runs CEM/MPC on dm_control
# cartpole-swingup for both the RSSM B1 baseline and the mechanics B_main
# checkpoint, across:
#   - 1 nominal condition
#   - 5 physical-OOD conditions (mass / pole-length scales from research spec)
#   - 5 visual-OOD conditions (color gain + brightness shifts)
#
# Each eval-control invocation: ~5–15 min on MPS. Full grid (11 × 2 = 22
# runs) ≈ 2–4 h total. Outputs land at:
#   eval/control-grid/{B1,B_main}/{regime}/returns.json
#   eval/control-grid/{B1,B_main}/{regime}/summary.csv
#
# Prerequisites:
#   - checkpoints/rssm-phase0-reward/latest.pt   (from train_rssm_with_reward.sh)
#   - checkpoints/mechanics-phase3-reward/latest.pt
#                                                (from train_mechanics_with_reward.sh)

set -euo pipefail

EPISODES=${EPISODES:-10}
HORIZON=${HORIZON:-15}
N_SAMPLES=${N_SAMPLES:-512}
N_ITERS=${N_ITERS:-3}

declare -A CHECKPOINTS=(
    [B1]="checkpoints/rssm-phase0-reward/latest.pt"
    [B_main]="checkpoints/mechanics-phase3-reward/latest.pt"
)

# Each condition is "name|extra_args". An empty extra_args means the
# nominal env. Visual conditions use --color-gain-* / --brightness-*.
CONDITIONS=(
    "nominal|"
    "mass_pole_05|--physics-scale body_mass_2=0.5"
    "mass_pole_15|--physics-scale body_mass_2=1.5"
    "mass_pole_20|--physics-scale body_mass_2=2.0"
    "length_pole_05|--physics-scale pole_length=0.5"
    "length_pole_15|--physics-scale pole_length=1.5"
    "color_gain_07|--color-gain-low 0.7 --color-gain-high 0.7"
    "color_gain_13|--color-gain-low 1.3 --color-gain-high 1.3"
    "brightness_dim|--brightness-low -0.2 --brightness-high -0.2"
    "brightness_bright|--brightness-low 0.2 --brightness-high 0.2"
    "color_random|--color-gain-low 0.5 --color-gain-high 1.5"
)

for model in "${!CHECKPOINTS[@]}"; do
    ckpt="${CHECKPOINTS[$model]}"
    if [[ ! -f "$ckpt" ]]; then
        echo "WARN: $ckpt missing — skipping $model. Train it via the matching scripts/*_with_reward.sh." >&2
        continue
    fi
    for entry in "${CONDITIONS[@]}"; do
        name="${entry%%|*}"
        extra="${entry#*|}"
        out="eval/control-grid/${model}/${name}"
        echo "==> $model / $name"
        # shellcheck disable=SC2086
        .venv/bin/eval-control \
            --checkpoint-path "$ckpt" \
            --output-dir "$out" \
            --episodes "$EPISODES" \
            --horizon "$HORIZON" \
            --n-samples "$N_SAMPLES" \
            --n-iters "$N_ITERS" \
            $extra
    done
done

echo
echo "Grid complete. Aggregate with:"
echo "  python -c 'import json,glob;[print(p, json.load(open(p)).get(\"mean_return\")) for p in sorted(glob.glob(\"eval/control-grid/**/returns.json\",recursive=True))]'"
