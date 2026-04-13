# Model Documentation

This directory contains one concise model document per trainable model in the
project. Component modules such as encoders, decoders, and RSSM cells are covered
inside the document for the trainable model that owns them unless they become
standalone training targets later.

## Current Models

- [Minimal RSSM Visual World Model](rssm_visual_world_model.md): the current
  unstructured cartpole pixel world-model baseline.
- [Mechanics-Structured Visual World Model](mechanics_world_model.md): factored
  `(q, qdot, z_nuisance)` model with a learned Lagrangian + Rayleigh
  dissipation transition and a symplectic-style integrator.

## Shared Notation And Terms

| Symbol / term | Meaning |
| --- | --- |
| `B` | Batch size: number of independent sequences processed together. |
| `T` | Sequence length: number of timesteps in each training sample. |
| `C` | Number of image channels. The current pixel observations use `C = 3` for RGB. |
| `H`, `W` | Image height and width. The current model input is `84 x 84`. |
| `A` | Action dimension. For cartpole-swingup this is currently `A = 1`. |
| `E` | Encoder embedding size: dimensionality of the per-frame visual embedding. |
| `Z` | Stochastic latent size in the RSSM. |
| `h_t` | Deterministic recurrent hidden state at timestep `t`. |
| `z_t` | Stochastic latent state at timestep `t`. |
| `RSSM` | Recurrent state-space model: a latent dynamics model with deterministic recurrent state and stochastic latent state. |
| `obs_t` | Observation at timestep `t`; in this project this is an RGB image tensor after preprocessing. |
| `obs_{t+1}` | Next observation reached after applying `action_t` to `obs_t`. |
| `next_observations` | Dataset tensor containing `obs_{t+1}` for each sampled transition, shaped `[T, C, H, W]` before batching and `[B, T, C, H, W]` in dataloaders. |
| `prior` | Dynamics prediction distribution, written here as `p(z_t \| h_t)`, before seeing the current observation embedding. |
| `posterior` | Inference distribution, written here as `q(z_t \| h_t, embed_t)`, after conditioning on the current observation embedding. |
| `KL` | Kullback-Leibler divergence. The current RSSM loss uses `KL(q \|\| p)` to align posterior latents with prior dynamics. |
| `stop-gradient` | A tensor is used as a target but gradients are blocked through it. The latent consistency target uses a stop-gradient encoding of `obs_{t+1}`. |
| `BYOL-style` | A prediction objective that matches a learned prediction to a stop-gradient target representation without explicit negative samples. |
| `foreground reconstruction` | Reconstruction loss weighted by a batch background-subtraction mask from `abs(obs_t - mean_batch_frame)` so salient non-background pixels contribute more than static background. |
| `transition reconstruction` | Next-frame pixel loss that decodes the RSSM prior feature after `action_t` and compares it to `obs_{t+1}` / `next_observations[:, t]`. |
| `foreground transition reconstruction` | Transition reconstruction weighted by the same foreground mask, applied to the `obs_{t+1}` target. |
| `dynamic reconstruction` | Reconstruction loss weighted by a motion mask from `abs(obs_{t+1} - obs_t)` so moving pixels contribute more than static background. |
| `open-loop` | Prediction mode where the model stops using future observations after a warmup/context window and rolls forward using latents and actions only. |
| `reconstruction` | Same-timestep posterior decode: target `obs_t` is compared against decoded `recon_t`. |
