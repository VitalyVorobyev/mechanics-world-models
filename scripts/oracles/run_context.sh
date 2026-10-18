#!/usr/bin/env bash
# Context oracle (H1b matched baseline), seeds in parallel. Resumable: re-run the same command.
# Extra arguments go to context_oracle.py, e.g.
#   TAG=context_noise_0.3px scripts/oracles/run_context.sh --sim-shift cart_mass_x0.3=0.3 --pose-noise 0.01 0.5
# Progress: tail -f runs/oracles/logs/${TAG}_seed*.log   ("[cells done/total]" lines)
# Stop:     pkill -f context_oracle.py   (cached pretraining/cells survive)
set -euo pipefail
cd "$(dirname "$0")/../.."
SEEDS=${SEEDS:-"0 1 2"}
THREADS=${THREADS:-4}
TAG=${TAG:-context}
mkdir -p runs/oracles/logs
pids=()
for seed in $SEEDS; do
  .venv/bin/python -u scripts/oracles/context_oracle.py --seed "$seed" --threads "$THREADS" "$@" \
    >> "runs/oracles/logs/${TAG}_seed${seed}.log" 2>&1 &
  pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit $status
