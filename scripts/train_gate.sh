#!/bin/bash
# Gate experiment (October 2026 review, docs/current_status.md "Gate").
#
# Trains the mechanics model with the pose forced through q:
#   --nuisance-mode style    decoder renders from q only; z = per-channel colour
#   --angle-dims 1           pole angle lives on S^1, seen as (cos, sin)
#   --qdot-stencil backward  causal q̇, the semi-implicit Euler convention
#   --dissipation-form linear  constant PSD damping, provably passive
#   dt omitted               resolves to 0.02 s from the dataset
#
# ARM selects the dynamics:
#   lagrangian  learned Euler-Lagrange dynamics (B_main)
#   mlp         unstructured q̈ = MLP(q, q̇, u), same encoder / bottleneck /
#               integrator. This is the control that isolates the Lagrangian.
#
# Usage: bash scripts/train_gate.sh lagrangian   (or: mlp)
# Expected wall-clock: ~10 h per arm on MPS for 10 epochs (Phase-3 took
# ~1.1-2.4 s/step); the mlp arm is faster.

set -euo pipefail

ARM=${1:?usage: train_gate.sh lagrangian|mlp}
case "$ARM" in
    lagrangian|mlp) ;;
    *) echo "ARM must be 'lagrangian' or 'mlp'" >&2; exit 1 ;;
esac

# exec: the trainer replaces this shell so SIGINT/SIGTERM reach it directly
# (it then writes a resumable latest.pt and exits with code 130).
exec .venv/bin/train-mechanics \
    --dataset-dir data/cartpole-swingup-random-100k-v2 \
    --checkpoint-dir "checkpoints/gate-${ARM}" \
    --sequence-length 32 --batch-size 32 --epochs 10 \
    --q-dim 2 --nuisance-dim 4 \
    --angle-dims 1 \
    --nuisance-mode style \
    --qdot-stencil backward \
    --dynamics-form "$ARM" \
    --dissipation-form linear \
    --imagination-context-steps 4 --imagination-horizon 12 \
    --foreground-imagination-reconstruction-weight 2.0 \
    --kl-weight 0.1 \
    --kl-free-nats-mech 0.5 --kl-free-nats-nuisance 3.0 \
    --nuisance-min-std 1.0 \
    --learning-rate 1e-4 --warmup-steps 2000 \
    --reward-head --reward-weight 1.0 --imagined-reward-weight 1.0 \
    --auto-resume \
    --checkpoint-every-steps 250 \
    --log-every-steps 50 \
    --val-open-loop-every-steps 500 \
    --val-open-loop-warmup 4 --val-open-loop-horizons 1,5,10,20
