.venv/bin/eval-mechanics \
    --checkpoint-path checkpoints/mechanics-phase2/latest.pt \
    --dataset-dir data/cartpole-swingup-random-100k \
    --output-dir eval/mechanics-phase2 \
    --sequence-length 32 \
    --warmup-length 4 \
    --horizons 1 5 10 20 \
    --num-sequences 64 \
    --num-visualizations 4
