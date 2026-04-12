# Minimal RSSM Visual World Model

## Purpose

This is the current trainable visual world-model baseline for cartpole pixel
experiments. It learns an unstructured recurrent latent dynamics model from
offline random-policy trajectories, reconstructs image observations for
diagnostics, and can optionally add a transition-aligned latent consistency
objective.

The model exists to provide a simple reference point before adding the
mechanics-structured model. It is not a Dreamer reproduction: there is no reward
head, value head, actor, policy learning, MPC, or planning loop in this model.

## Architecture And Parameters

Implementation entry point: `src/models/world_model.py::VisualWorldModel`.
Shared notation such as `B`, `T`, `E`, `Z`, prior, posterior, and open-loop is
defined in the [model documentation index](README.md#shared-notation-and-terms).

Default cartpole training configuration:

- observations: `[B, T, 3, 84, 84]`, float32 in `[0, 1]`
- actions: `[B, T, 1]`, float32
- encoder embedding size `E = 256`
- deterministic RSSM hidden size `H = 128`
- stochastic RSSM latent size `Z = 32`
- decoder output: `[B, T, 3, 84, 84]`, float32 in `[0, 1]`
- latent predictor output: `[B, T, E]`

Parameter count for this default configuration:

| Module | Parameters |
| --- | ---: |
| Convolutional encoder | 1,247,328 |
| RSSM dynamics | 144,896 |
| Image decoder | 942,947 |
| Latent predictor | 107,008 |
| **Total** | **2,442,179** |

Parameter counts change if `action_dim`, `embedding_size`, `hidden_size`, or
`latent_size` changes.

### Encoder

`ConvEncoder` applies four stride-2 convolution blocks with channels
`3 -> 32 -> 64 -> 128 -> 128`, flattens the resulting `5 x 5` feature map, and
projects it to an embedding vector. It maps each frame sequence from
`[B, T, 3, 84, 84]` to `[B, T, E]`.

### RSSM Dynamics

`RSSM` maintains:

- deterministic recurrent state `h_t` with size `H`
- stochastic latent state `z_t` with size `Z`
- action-conditioned transition through a `GRUCell`
- diagonal Gaussian prior `p(z_t | h_t)`
- diagonal Gaussian posterior `q(z_t | h_t, embed_t)`

During posterior training/inference, `h_t` is the deterministic state aligned to
`obs_t`. At `t = 0` it is initialized to zero. For later steps it has already
been advanced by `z_{t-1}` and `action_{t-1}`. The model computes
`p(z_t | h_t)`, then conditions the posterior on the current image embedding.
Transition prediction applies `action_t` to the posterior state at `obs_t` and
targets `obs_{t+1}`. Evaluation utilities also expose deterministic mean-based
posterior reconstruction and open-loop prior rollout paths.

### Decoder

`ImageDecoder` takes RSSM features `[h_t, z_t]` shaped `[B, T, H + Z]`, projects
them to a small spatial feature map, applies transposed convolutions, and returns
sigmoid-normalized RGB reconstructions shaped `[B, T, 3, 84, 84]`.

### Latent Predictor

`VisualWorldModel.latent_predictor` is a small MLP from predicted next RSSM
prior features `[B, T, H + Z]` to encoder embedding space `[B, T, E]`. It is
used only when the dataset batch provides `next_observations`.

The transition-aligned path treats saved data as
`obs_t --action_t--> obs_{t+1}`:

1. encode `obs_t` as `[B, T, E]`
2. infer the current posterior latent from `obs_t`
3. apply `action_t` through the RSSM recurrent transition
4. predict the prior feature for `obs_{t+1}`
5. map that feature to the encoder embedding space and compare it to a
   stop-gradient encoding of `obs_{t+1}`
6. optionally decode that predicted prior feature and compare it directly to
   `obs_{t+1}` with a transition reconstruction loss

## Objective Function

Training uses the loss in `src/models/losses.py::compute_world_model_loss`:

```text
total_loss =
    reconstruction_weight * reconstruction_loss
  + foreground_reconstruction_weight * foreground_reconstruction_loss
  + transition_reconstruction_weight * transition_reconstruction_loss
  + foreground_transition_reconstruction_weight * foreground_transition_reconstruction_loss
  + dynamic_reconstruction_weight * dynamic_reconstruction_loss
  + kl_weight * kl_loss
  + latent_consistency_weight * latent_consistency_loss
```

where:

- `reconstruction_loss` is full-frame pixel MSE between decoder output and the
  same-timestep target observation, both shaped `[B, T, 3, 84, 84]`
- `foreground_reconstruction_loss` is pixel MSE weighted by a normalized
  background-subtraction mask from `abs(obs_t - mean_batch_frame)`; the mask is
  max-pooled around salient pixels, has a configurable static-pixel floor, and is
  normalized to mean one per frame
- `dynamic_reconstruction_loss` is pixel MSE weighted by a normalized motion mask
  derived from `abs(obs_{t+1} - obs_t)`; the mask is max-pooled around moving
  pixels, has a configurable static-pixel floor, and is normalized to mean one
  per frame
- `transition_reconstruction_loss` decodes the transition prior feature after
  `action_t` and compares it to `obs_{t+1}` / `next_observations[:, t]`, not to
  same-index `obs_t`
- `foreground_transition_reconstruction_loss` applies the same normalized
  background-subtraction mask to the transition reconstruction target
  `obs_{t+1}`; this directly trains `decoder(next_prior_features)` on the
  foreground-heavy next-frame objective
- `kl_loss` is `KL(q(z_t | h_t, embed_t) || p(z_t | h_t))` for diagonal
  Gaussian posterior and prior, summed over latent dimensions and averaged over
  batch and time
- `latent_consistency_loss = mean(1 - cosine_similarity(predicted_next_embedding,
  stopgrad(encoded_obs_{t+1})))`
- `weighted_reconstruction_loss = reconstruction_weight * reconstruction_loss`
- `weighted_foreground_reconstruction_loss = foreground_reconstruction_weight *
  foreground_reconstruction_loss`
- `weighted_dynamic_reconstruction_loss = dynamic_reconstruction_weight *
  dynamic_reconstruction_loss`
- `weighted_transition_reconstruction_loss = transition_reconstruction_weight *
  transition_reconstruction_loss`
- `weighted_foreground_transition_reconstruction_loss =
  foreground_transition_reconstruction_weight *
  foreground_transition_reconstruction_loss`
- `weighted_kl_loss = kl_weight * kl_loss`
- `weighted_latent_consistency_loss = latent_consistency_weight *
  latent_consistency_loss`

Default CLI weights preserve the original pixel/KL baseline:
`reconstruction_weight = 1.0`, `kl_weight = 1.0`, and
`latent_consistency_weight = 0.0`, with `foreground_reconstruction_weight = 0.0`,
`transition_reconstruction_weight = 0.0`,
`foreground_transition_reconstruction_weight = 0.0`, and
`dynamic_reconstruction_weight = 0.0`. The current transition-prior diagnostic
keeps posterior foreground reconstruction enabled and adds an opt-in
foreground transition reconstruction term so the prior path is trained directly
against `obs_{t+1}`.

The current objective intentionally does not include reward prediction,
hand-authored foreground segmentation, InfoNCE negatives, augmentation, or a
reconstruction-free B4/MuDreamer-style contrastive baseline. Recent diagnostics
suggest that full-frame pixel MSE can over-reward static background
reconstruction relative to the small moving cartpole foreground, so the
foreground reconstruction, transition reconstruction, dynamic reconstruction,
and latent consistency losses are auxiliary objectives for the existing RSSM
baseline.

## Optimizer

The training entry point is `src/train/train_rssm.py`.

Default optimizer settings:

- optimizer: `torch.optim.Adam`
- learning rate: CLI `--learning-rate`, default `1e-3`
- gradient clipping: CLI `--grad-clip`, default `100.0`
- batch size, sequence length, loss weights, and model sizes are all
  CLI-configured
- checkpoints contain model state, optimizer state, config, epoch, global step,
  latest metrics, and history path

There is currently no scheduler, KL warmup, free-nats, mixed precision, or
distributed training.

## Relation To Other Models

This model is the unstructured baseline. Future mechanics-structured models
should be compared against it on the same datasets, sequence lengths, action
repeat, observation preprocessing, metrics, and compute budget.

The intended next model family will factor the latent state into mechanical
coordinates and nuisance variables, for example `(q, qdot, z_nuisance)`, and use
learned Lagrangian dynamics plus Rayleigh dissipation. That future model may
reuse the dataset pipeline, evaluation utilities, and possibly parts of the
encoder/decoder, but it should have its own model document.
