.venv/bin/eval-rssm \
  --checkpoint-path checkpoints/rssm-phase0/epoch_0005.pt \
  --dataset-dir data/cartpole-swingup-random-100k \
  --output-dir eval/rssm-phase0 \
  --sequence-length 32 \
  --warmup-length 5 \
  --horizons 1 5 10
