#!/bin/bash
# Collect four mildly-perturbed cartpole datasets for the joint OOD training
# experiment that tests the gauge-freedom hypothesis behind Phase-3's K2
# failure. See docs/current_status.md for context.
#
# Training regimes are kept mild (×0.7 / ×1.3) so the research-spec OOD grid
# (mass ×{0.5, 1.5, 2.0}, length ×{0.5, 1.5}) still has held-out points.
#
# Expected wall-clock: ~30-50 minutes on MPS for 4 × 15k frames.
#
# v2 (October 2026): the April datasets (no -v2 suffix) are physically wrong.
# Their "length" runs scaled geom_size_2_1, which is the static rail `rail2`,
# so they only changed a drawn rail. Their mass runs scaled body_mass_2
# without the pole inertia. These v2 runs use pole_length (MJCF-level, so
# inertia and centre of mass are recompiled) and inertia-consistent mass
# scaling. Do not mix v1 and v2 regimes.

set -euo pipefail

FRAMES=15000
ACTION_REPEAT=2
IMAGE_SIZE=84

.venv/bin/collect-cartpole \
    --output-dir data/cartpole-ood-mass07-15k-v2 \
    --total-frames "$FRAMES" \
    --seed 1 \
    --action-repeat "$ACTION_REPEAT" \
    --image-size "$IMAGE_SIZE" \
    --env-name cartpole-swingup \
    --physics-scale body_mass_2=0.7

.venv/bin/collect-cartpole \
    --output-dir data/cartpole-ood-mass13-15k-v2 \
    --total-frames "$FRAMES" \
    --seed 2 \
    --action-repeat "$ACTION_REPEAT" \
    --image-size "$IMAGE_SIZE" \
    --env-name cartpole-swingup \
    --physics-scale body_mass_2=1.3

.venv/bin/collect-cartpole \
    --output-dir data/cartpole-ood-length07-15k-v2 \
    --total-frames "$FRAMES" \
    --seed 3 \
    --action-repeat "$ACTION_REPEAT" \
    --image-size "$IMAGE_SIZE" \
    --env-name cartpole-swingup \
    --physics-scale pole_length=0.7

.venv/bin/collect-cartpole \
    --output-dir data/cartpole-ood-length13-15k-v2 \
    --total-frames "$FRAMES" \
    --seed 4 \
    --action-repeat "$ACTION_REPEAT" \
    --image-size "$IMAGE_SIZE" \
    --env-name cartpole-swingup \
    --physics-scale pole_length=1.3
