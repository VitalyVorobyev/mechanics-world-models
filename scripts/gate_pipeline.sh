#!/bin/bash
# Resumable driver for the October 2026 gate experiment.
#
#   bash scripts/gate_pipeline.sh start    # launch in the background (nohup)
#   bash scripts/gate_pipeline.sh status   # progress of every stage + ETA
#   bash scripts/gate_pipeline.sh stop     # graceful stop; training saves latest.pt
#   bash scripts/gate_pipeline.sh run      # run in the foreground (Ctrl-C = stop)
#
# `start` after a `stop` (or a crash / reboot) continues where it left off:
# finished stages are skipped via marker files in runs/gate/, training
# resumes from its last mid-epoch checkpoint (written every 250 steps and on
# stop), and a partially collected dataset is deleted and recollected.
#
# Stages (in order):
#   collect-<regime>   OOD-v2 datasets (physically consistent perturbations)
#   train-lagrangian   scripts/train_gate.sh lagrangian
#   train-mlp          scripts/train_gate.sh mlp
#   eval               scripts/eval_gate.sh (both arms + rssm-phase0-reward)
#   report             scripts/gate_report.py -> runs/gate/report.md
#
# Logs: runs/gate/<stage>.log and runs/gate/pipeline.log.

set -euo pipefail

cd "$(dirname "$0")/.."
RUN_DIR=runs/gate
PID_FILE="$RUN_DIR/pipeline.pid"
mkdir -p "$RUN_DIR"

# regime | seed | physics scale. Mild shifts (x0.7 / x1.3) plus two stronger
# extrapolation points; no model is trained on any of them.
REGIMES=(
    "mass07|1|body_mass_2=0.7"
    "mass13|2|body_mass_2=1.3"
    "length07|3|pole_length=0.7"
    "length13|4|pole_length=1.3"
    "mass20|5|body_mass_2=2.0"
    "length15|6|pole_length=1.5"
)
OOD_FRAMES=15000

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$RUN_DIR/pipeline.log"; }

CHILD=""
on_signal() {
    # SIGTERM, not SIGINT: background jobs of a non-interactive shell start
    # with SIGINT ignored. The trainer turns SIGTERM into "checkpoint and exit".
    log "stop requested; forwarding SIGTERM to the running stage"
    if [[ -n "$CHILD" ]] && kill -0 "$CHILD" 2>/dev/null; then
        kill -TERM "$CHILD" 2>/dev/null || true
        wait "$CHILD" 2>/dev/null || true
    fi
    log "pipeline stopped; resume with: bash scripts/gate_pipeline.sh start"
    rm -f "$PID_FILE"
    exit 130
}

# run_stage <name> <command...>: skip if done, otherwise run with a log and
# mark done on exit code 0. A non-zero exit stops the pipeline.
run_stage() {
    local name=$1
    shift
    if [[ -f "$RUN_DIR/$name.done" ]]; then
        log "skip $name (done)"
        return 0
    fi
    log "start $name: $*"
    "$@" >>"$RUN_DIR/$name.log" 2>&1 &
    CHILD=$!
    local status=0
    wait "$CHILD" || status=$?
    CHILD=""
    if [[ $status -ne 0 ]]; then
        log "stage $name exited with status $status (see $RUN_DIR/$name.log)"
        rm -f "$PID_FILE"
        exit "$status"
    fi
    touch "$RUN_DIR/$name.done"
    log "done $name"
}

collect_regime() {
    local regime=$1 seed=$2 scale=$3
    local out="data/cartpole-ood-${regime}-15k-v2"
    # Only ever delete our own partial v2 output from an interrupted attempt.
    if [[ -d "$out" ]]; then
        rm -rf "$out"
    fi
    # exec: this background subshell becomes the collector, so the stop
    # signal reaches it directly.
    exec .venv/bin/collect-cartpole \
        --output-dir "$out" \
        --total-frames "$OOD_FRAMES" \
        --seed "$seed" \
        --action-repeat 2 \
        --image-size 84 \
        --env-name cartpole-swingup \
        --physics-scale "$scale"
}

run_pipeline() {
    echo $$ >"$PID_FILE"
    trap on_signal INT TERM
    log "pipeline started (pid $$)"
    for entry in "${REGIMES[@]}"; do
        IFS='|' read -r regime seed scale <<<"$entry"
        run_stage "collect-$regime" collect_regime "$regime" "$seed" "$scale"
    done
    run_stage train-lagrangian bash scripts/train_gate.sh lagrangian
    run_stage train-mlp bash scripts/train_gate.sh mlp
    run_stage eval bash scripts/eval_gate.sh
    run_stage report .venv/bin/python scripts/gate_report.py --output "$RUN_DIR/report.md"
    log "pipeline finished; see $RUN_DIR/report.md"
    rm -f "$PID_FILE"
}

pipeline_pid() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null && cat "$PID_FILE"
}

case "${1:-}" in
    run)
        if pid=$(pipeline_pid); then echo "already running (pid $pid)" >&2; exit 1; fi
        run_pipeline
        ;;
    start)
        if pid=$(pipeline_pid); then echo "already running (pid $pid)"; exit 0; fi
        nohup bash "$0" run >>"$RUN_DIR/pipeline.out" 2>&1 &
        sleep 1
        echo "started (pid $(cat "$PID_FILE" 2>/dev/null || echo '?')); status: bash $0 status"
        ;;
    stop)
        if pid=$(pipeline_pid); then
            kill -TERM "$pid"
            echo "stop signal sent to pipeline pid $pid; waiting for the stage to checkpoint..."
            while kill -0 "$pid" 2>/dev/null; do sleep 2; done
            echo "stopped"
        else
            echo "not running"
        fi
        ;;
    status)
        .venv/bin/python scripts/gate_status.py
        ;;
    *)
        sed -n '2,22p' "$0"
        exit 2
        ;;
esac
