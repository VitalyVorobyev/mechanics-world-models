#!/usr/bin/env bash
# H1b dynamics oracle v2, seeds in parallel. Resumable: re-run the same command.
# Progress: tail -f runs/oracles/logs/${TAG:-dynamics_v2}_seed*.log   (TAG names the log)
# Stop:     pkill -f dynamics_oracle.py   (cached pretraining/cells survive)
set -euo pipefail
cd "$(dirname "$0")/../.."
SEEDS=${SEEDS:-"0 1 2"}
THREADS=${THREADS:-4}
TAG=${TAG:-dynamics_v2}
mkdir -p runs/oracles/logs
pids=()
for seed in $SEEDS; do
  .venv/bin/python -u scripts/oracles/dynamics_oracle.py \
    --shift cart_mass_x05=data/cartpole-ood-cartmass05-15k-v2 \
    --shift cart_mass_x2=data/cartpole-ood-cartmass20-15k-v2 \
    --seed "$seed" --threads "$THREADS" "$@" \
    >> "runs/oracles/logs/${TAG}_seed${seed}.log" 2>&1 &
  pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit $status
