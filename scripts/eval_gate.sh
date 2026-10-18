#!/bin/bash
# Evaluate the gate-experiment arms against the RSSM baseline on identical
# sequences (same seed / val split / 256 windows), in-distribution and on the
# physically consistent OOD datasets.
#
# Prerequisites:
#   checkpoints/gate-lagrangian/latest.pt, checkpoints/gate-mlp/latest.pt
#     (bash scripts/train_gate.sh lagrangian|mlp)
#   checkpoints/rssm-phase0-reward/latest.pt          (already on disk)
#   data/cartpole-ood-*-15k-v2/  (bash scripts/collect_cartpole_ood.sh) — optional;
#     OOD datasets that do not exist yet are skipped.
#
# Gate (docs/current_status.md): continue only if, for gate-lagrangian,
#   1. state_probes.q->pose.mlp R² > 0.8 on the nominal val split,
#   2. foreground_open_loop_mse_h20 <= the RSSM's on the same sequences, and
#   3. its h20 degradation on the OOD-v2 sets is smaller than the RSSM's,
# and gate-lagrangian beats gate-mlp on 2-3 (otherwise the Lagrangian
# structure is not what helps).
#
# Expected wall-clock: ~1 min per (model, dataset) on CPU.

set -euo pipefail

COMMON=(--sequence-length 32 --warmup-length 5 --horizons 1 5 10 20
        --num-sequences 256 --num-visualizations 2 --device cpu)
DATASETS=(
    data/cartpole-swingup-random-100k-v2
    data/cartpole-ood-mass07-15k-v2
    data/cartpole-ood-mass13-15k-v2
    data/cartpole-ood-length07-15k-v2
    data/cartpole-ood-length13-15k-v2
    data/cartpole-ood-mass20-15k-v2
    data/cartpole-ood-length15-15k-v2
)

for dataset in "${DATASETS[@]}"; do
    if [[ ! -d "$dataset" ]]; then
        echo "skip $dataset (not collected yet)"
        continue
    fi
    name=$(basename "$dataset")
    # No model trained on the OOD sets, so their (larger) train split is
    # unseen data too and gives the probes more episodes.
    split=val
    [[ "$name" == cartpole-ood-* ]] && split=train
    for arm in lagrangian mlp; do
        .venv/bin/eval-mechanics \
            --checkpoint-path "checkpoints/gate-${arm}/latest.pt" \
            --dataset-dir "$dataset" \
            --output-dir "eval/gate/${arm}/${name}" \
            --split "$split" \
            "${COMMON[@]}"
    done
    .venv/bin/eval-rssm \
        --checkpoint-path checkpoints/rssm-phase0-reward/latest.pt \
        --dataset-dir "$dataset" \
        --output-dir "eval/gate/rssm/${name}" \
        --split "$split" \
        "${COMMON[@]}"
done
