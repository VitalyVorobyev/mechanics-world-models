.venv/bin/eval-mechanics \
    --checkpoint-path checkpoints/mechanics-phase3-finitediff/latest.pt \
    --dataset-dir data/cartpole-swingup-random-100k-v2 \
    --output-dir eval/mechanics-phase3-finitediff \
    --sequence-length 32 \
    --warmup-length 5 \
    --horizons 1 5 10 \
    --num-sequences 64 \
    --num-visualizations 4
