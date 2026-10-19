#!/bin/bash
# Evaluate the joint-trained mechanics model:
#   1) on the joint val split — primary K1 + K2 signal, mixes all training regimes.
#   2) per-regime, on each OOD training regime — checks that the learned q is
#      physical across regimes (the actual gauge-breaking test).
#   3) optionally on held-out OOD (see comment block at the bottom) — tests
#      extrapolation beyond training perturbations.

set -euo pipefail

CKPT=checkpoints/mechanics-joint-v3/latest.pt
COMMON_ARGS=(--sequence-length 32 --warmup-length 5 --horizons 1 5 10)

# 1) joint-dataset eval (primary K1/K2)
.venv/bin/eval-mechanics \
    --checkpoint-path "$CKPT" \
    --dataset-dir data/cartpole-joint-v3 \
    --output-dir eval/mechanics-joint-v3 \
    "${COMMON_ARGS[@]}" \
    --num-sequences 64 \
    --num-visualizations 4

# 2) per-regime evals (nominal + 4 training-OOD regimes)
for regime_dir in \
    data/cartpole-swingup-random-100k-v2 \
    data/cartpole-ood-mass07-15k \
    data/cartpole-ood-mass13-15k \
    data/cartpole-ood-length07-15k \
    data/cartpole-ood-length13-15k
do
    name=$(basename "$regime_dir")
    .venv/bin/eval-mechanics \
        --checkpoint-path "$CKPT" \
        --dataset-dir "$regime_dir" \
        --output-dir "eval/mechanics-joint-v3-${name}" \
        "${COMMON_ARGS[@]}" \
        --num-sequences 32 \
        --num-visualizations 1
done

# 3) held-out OOD (uncomment after collecting them):
#   .venv/bin/collect-cartpole --physics-scale body_mass_2=0.5  ...
#   .venv/bin/collect-cartpole --physics-scale body_mass_2=2.0  ...
#   .venv/bin/collect-cartpole --physics-scale pole_length=0.5 ...
#   .venv/bin/collect-cartpole --physics-scale pole_length=1.5 ...
# These are the research-spec shift targets for H1; defer until K2 passes on (1).
