# Current Status and Open Problem

This project is currently at the unstructured-baseline stage. The code can
collect pixel observations from DeepMind Control `cartpole-swingup`, train a
minimal RSSM visual world model offline, and generate reconstruction and
open-loop prediction diagnostics. The mechanics-structured model is still ahead
of us.

## What Exists

The current baseline includes:

- random-policy cartpole pixel collection with MuJoCo offscreen rendering
- one-episode-per-file `.npz` storage with `obs_t --action_t--> obs_{t+1}`
  alignment
- a PyTorch sequence dataset that returns `observations`, `next_observations`,
  `actions`, `rewards`, and `dones`
- a compact RSSM-style visual world model with encoder, RSSM transition,
  decoder, and optional latent consistency head
- an offline trainer with checkpoints, JSONL history, rich terminal logging, and
  loss plotting
- evaluation utilities for posterior reconstruction, one-step prediction, and
  multi-step open-loop prediction
- debugging utilities for reconstruction grids, one-batch overfitting, dataset
  image statistics, crop previews, dynamic masks, and foreground masks

The important positive result is that the reconstruction path is capable. In the
fixed-batch and fixed-sequence overfit diagnostics, the model can memorize and
reconstruct the cartpole. That makes a basic decoder or tensor-alignment bug less
likely.

## The Current Problem

The first normal RSSM training runs reconstructed the static background much
better than the cartpole foreground. Reconstructions and open-loop predictions
often contained only a faint cartpole silhouette. That made the baseline a poor
comparison target for the future mechanics-structured model.

The first working diagnosis was objective imbalance. The cartpole occupies a small
fraction of the 84x84 image, while the static background occupies most pixels.
Plain full-frame pixel MSE can therefore improve by modeling the background well
and only weakly modeling the moving foreground. This is consistent with the
observed behavior:

- posterior reconstruction overfit works on a fixed batch or fixed sequence
- normal training favors background reconstruction
- center cropping did not fix the issue in the tested setup
- latent consistency did not remove the faint-foreground failure mode
- dynamic reconstruction masks were not ideal because they often emphasized a
  right-edge/background artifact rather than the cartpole

The latest experiments changed that picture. A strong foreground reconstruction
stress run with `foreground_reconstruction_weight=20`, `reconstruction_weight=0`,
and very small KL fixed posterior reconstruction: qualitative reconstructions are
sharp, and reconstruction MSE is around `3e-4`. Adding `kl_weight=1e-5` and
`latent_consistency_weight=0.01` kept posterior reconstruction sharp and drove
the latent consistency loss down, but open-loop predictions still drifted toward
background-only frames after the warmup window.

This means the posterior reconstruction path is no longer the main blocker. The
current problem is the prior transition path. The saved data is
`obs_t --action_t--> obs_{t+1}`, so transition predictions must be trained and
measured against `next_observations[:, t]`, not the same-index
`observations[:, t]`. Any off-by-one action/state convention can look like a
weak dynamics model even when posterior reconstructions are good.

After the alignment patch, posterior reconstruction remains good, but decoded
transition predictions are still weak and background-heavy. The likely remaining
issue is that the decoder has been trained mostly on posterior features
`[h_t, z_t]`, while qualitative prediction videos decode transition prior
features for `obs_{t+1}`. The latent predictor can reduce cosine loss through
its own MLP without forcing `decoder(next_prior_features)` to produce a sharp
cartpole frame.

## Next Investigation

The next patch adds an explicit transition reconstruction diagnostic/objective:

- posterior reconstruction at index `t` still targets `obs_t`
- transition reconstruction at index `t` decodes the prior feature after
  `action_t` and targets `obs_{t+1}`
- the foreground-masked transition reconstruction term should directly train the
  prior/decoder path used by one-step and open-loop videos
- scalar eval should continue to include foreground-masked prediction MSE,
  because full-frame MSE hides foreground failure

Only after that alignment check should we move to a reconstruction-free or
target-encoder latent baseline. The current evidence says latent prediction alone
is not the next clean move, because decoded transition-prior features are still
undertrained.

## What This Means for the Research Plan

This is still a baseline debugging problem, not the final research question. The
structured mechanics model should not be added until the RSSM baseline has a
reasonable in-distribution visual prediction path. Otherwise, any comparison
against the future Lagrangian or dissipative model would mix up two issues:
mechanical structure and an under-debugged pixel objective.
