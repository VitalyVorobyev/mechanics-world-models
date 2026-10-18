#!/bin/bash
# Assemble a joint cartpole dataset by symlinking per-regime episode files
# into a single directory. The sequence loader globs episode_*.npz, so all
# regime files must preserve that prefix and have globally-unique names.
#
# The seeded val-split (src/data/sequence_dataset.py::split_episode_files)
# will mix regimes into val because filenames sort by regime but the RNG
# permutes across the full pool.
#
# Prerequisites:
#   - data/cartpole-swingup-random-100k-v2/       (nominal; from collect_cartpole.sh)
#   - data/cartpole-ood-mass07-15k/
#   - data/cartpole-ood-mass13-15k/
#   - data/cartpole-ood-length07-15k/
#   - data/cartpole-ood-length13-15k/              (from collect_cartpole_ood.sh)

set -euo pipefail

JOINT=data/cartpole-joint-v3
mkdir -p "$JOINT"

# Wipe prior symlinks to avoid stale references (only removes symlinks, not
# accidental real files).
find "$JOINT" -maxdepth 1 -type l -name 'episode_*.npz' -delete

link_regime() {
    local src=$1
    local regime=$2
    if [[ ! -d "$src" ]]; then
        echo "ERROR: $src does not exist. Run scripts/collect_cartpole.sh / collect_cartpole_ood.sh first." >&2
        return 1
    fi
    local count=0
    for f in "$src"/episode_*.npz; do
        [[ -e "$f" ]] || continue
        local base
        base=$(basename "$f" .npz)                # episode_0001
        local idx=${base#episode_}                # 0001
        ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" \
               "$JOINT/episode_${regime}_${idx}.npz"
        count=$((count + 1))
    done
    echo "  linked $count episode(s) from $src as regime '$regime'"
}

link_regime data/cartpole-swingup-random-100k-v2 nominal
link_regime data/cartpole-ood-mass07-15k          mass07
link_regime data/cartpole-ood-mass13-15k          mass13
link_regime data/cartpole-ood-length07-15k        length07
link_regime data/cartpole-ood-length13-15k        length13

total=$(find "$JOINT" -maxdepth 1 -name 'episode_*.npz' | wc -l | tr -d ' ')
echo
echo "Joint dataset at $JOINT contains $total episode file(s)."
