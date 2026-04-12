.venv/bin/foreground-mask-preview \
  --dataset-dir data/cartpole-swingup-random-100k \
  --output-path eval/foreground-mask-preview-fg2.png \
  --sequence-length 32 \
  --batch-size 32 \
  --foreground-mask-floor 0.02 \
  --foreground-mask-kernel-size 7