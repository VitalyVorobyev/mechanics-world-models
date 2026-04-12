.venv/bin/debug-reconstruction \
  --checkpoint-path checkpoints/rssm-cartpole-aligned-fg20-kl1e5-latent001/latest.pt \
  --dataset-dir data/cartpole-swingup-random-100k \
  --output-dir eval/debug-aligned-fg20-kl1e5-latent001 \
  --sequence-length 32 \
  --batch-size 4 \
  --timesteps 0 8 16 24 31
