.venv/bin/eval-rssm \
  --checkpoint-path checkpoints/rssm-cartpole-aligned-fg20-kl1e5-latent001/latest.pt \
  --dataset-dir data/cartpole-swingup-random-100k \
  --output-dir eval/rssm-cartpole-aligned-fg20-kl1e5-latent001 \
  --sequence-length 32 \
  --warmup-length 5 \
  --horizons 1 5 10
