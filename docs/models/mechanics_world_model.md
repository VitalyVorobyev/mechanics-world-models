# Mechanics-Structured Visual World Model

## Purpose

This is the Phase-2 model from the research roadmap: a **factored** latent
visual world model whose dynamics branch is a learned Lagrangian with Rayleigh
dissipation, integrated by a symplectic-style step. The factorization
`(q, qdot, z_nuisance)` is the structural inductive bias that the research
spec uses to test H1 (physical OOD), H2 (visual OOD), and H3 (dissipation
matters).

The model is trained **pixel-only** — no privileged simulator state is read
into any training loss. Stored ground-truth `qpos`/`qvel` from dm_control
exists in the dataset but is consumed only by the diagnostic linear probe
and energy-tracking metric in `src/eval/probes.py`.

The model exists alongside the
[unstructured RSSM baseline](rssm_visual_world_model.md) (B1). When run on
the same dataset and budget it answers the K1/K2 acceptance gates from the
research spec.

## Architecture and Parameters

Implementation entry point:
`src/models/mechanics/world_model.py::MechanicsWorldModel`. Shared notation
is the same as in the RSSM doc; mechanics-specific symbols below.

Default cartpole training configuration:

- observations: `[B, T, 3, 84, 84]`, float32 in `[0, 1]`
- actions: `[B, T, 1]`, float32
- generalized coordinate dim `d = 2` (cartpole)
- nuisance dim `k = 32`
- shared encoder embedding `E = 256`
- decoder input dim `2d + k = 36`
- integrator step `dt = 0.01` s, `n_substeps = 1`

### Encoder (`src/models/mechanics/encoder.py::FactoredEncoder`)

Shared `ConvEncoder` trunk (identical to the RSSM baseline) feeds two heads:

- **MechHead**: `Linear(E, 128) -> ELU -> Linear(128, 4d)` — emits
  `(q_mean, q_std, qdot_mean, qdot_std)` per frame, independent in time.
- **NuisanceHead**: `Linear(E, 128) -> ELU -> Linear(128, 2k)` — emits
  `(z_mean, z_std)` per frame.

Posterior stds use `softplus(raw) + min_std` with `mech_min_std = 0.05` and
`nuisance_min_std = 0.1`. The encoder is intentionally weak as a
factorizing module on its own; the gradient signal that forces the split
to mean anything comes from the Lagrangian prior + smoothness loss.

### Lagrangian Dynamics (`src/models/mechanics/lagrangian.py`)

Four small MLPs over `q`:

- `MassMatrix(q) -> M(q)`: lower-triangular Cholesky parameterization with
  `softplus`-positive diagonal and an `eps` ridge so `M = L Lᵀ + eps*I` is
  symmetric PD by construction.
- `Potential(q) -> V(q)`: scalar.
- `Dissipation(qdot) -> D = 0.5 qdotᵀ C(qdot) qdot` with PSD `C` (Cholesky).
  `dD/d(qdot)` is monotone in damping.
- `Actuation(q) -> B(q) ∈ ℝ^{d × a}`: generalized force from action `u`.

Forward dynamics use the **mechanical form** of Euler-Lagrange:

```
M(q) q̈ = dL/dq − Ṁ(q) q̇ − dD/d(qdot) + B(q) u
```

where `L = ½ q̇ᵀ M(q) q̇ − V(q)`. `dL/dq` is computed with autograd holding
`q̇` fixed; `Ṁ q̇` is computed as the forward JVP of `p(q) = M(q) q̇` along
the direction `q̇`. This is the closed-form identity for systems with
quadratic-in-`q̇` kinetic energy and avoids the generic double-Jacobian of a
free-form `L(q, q̇)`. Gradient flows back to all four primitives' parameters
during training.

### Integrator (`src/models/mechanics/integrator.py`)

Semi-implicit (symplectic) Euler:

```
qdot_{t+1} = qdot_t + h * q̈(q_t, qdot_t, u_t)
q_{t+1}    = q_t    + h * qdot_{t+1}
```

with `h = dt / n_substeps`. On a conservative (`D = 0`) Hamiltonian system
this is symplectic up to a bounded modified Hamiltonian, so energy drift
stays O(h²) rather than O(t) — the property `tests/test_mechanics_primitives.py`
verifies on a randomly initialized model.

### Transition (`src/models/mechanics/transition.py::MechanicsTransition`)

Composes the Lagrangian step on `(q, qdot)` with an AR(1) nuisance:

```
q_{t+1}, qdot_{t+1} = SemiImplicitEuler(M, V, D, B, q_t, qdot_t, u_t)
z_{t+1}             = alpha * z_t                # alpha learnable in (0, 1)
```

with three learnable per-coordinate prior stds (`q`, `qdot`, `z`). At `t=0`
the prior is unit Gaussian (no history to condition on).

### Decoder

`ImageDecoder` is identical to the RSSM baseline's; it consumes a
`[B, T, 2d + k]` feature tensor.

### Reward Head (optional)

When `--reward-head` is set, a `Linear -> ELU -> Linear -> ELU -> Linear`
MLP from features to a scalar reward. Required for MPC and the
research-spec primary metric. Disabled by default.

## Objective Function

Training uses
`src/models/mechanics/losses.py::compute_mechanics_world_model_loss`:

```
total_loss =
    reconstruction_weight * reconstruction_loss
  + foreground_reconstruction_weight * foreground_reconstruction_loss
  + imagination_reconstruction_weight * imagination_reconstruction_loss
  + foreground_imagination_reconstruction_weight * foreground_imagination_reconstruction_loss
  + kl_weight * (kl_q_loss + kl_qdot_loss + kl_z_loss)
  + smoothness_weight * smoothness_loss
  + reward_weight * reward_loss
  + imagined_reward_weight * imagined_reward_loss
```

Notable terms specific to this model:

- **Three-branch balanced KL** with a per-branch free-nats floor:
  `kl_free_nats_mech = 0.5` for `q` and `qdot`; `kl_free_nats_nuisance = 3.0`
  for `z`. The asymmetric budget keeps the nuisance branch loose so the
  encoder can park appearance there, while the mechanics branch is held
  tight against the integrator-derived prior. Each branch uses
  DreamerV2-style balancing with `alpha = 0.8`.
- **Smoothness prior on qdot**:
  `‖0.5 * (qdot_t + qdot_{t+1}) − stop_grad((q_{t+1} − q_t) / dt)‖²`. This is
  the single most informative regularizer against state collapse — it
  forces the encoder's two mechanics heads into a finite-difference
  derivative relationship, which is the assumption the Lagrangian step
  expects. Default weight `1.0`.
- **Imagination rollout** is identical in shape to the Phase-0 RSSM
  imagination contract: a posterior context of `K_ctx` steps followed by
  `K_imag` closed-loop prior steps decoded with the same decoder.
  `imagined_reconstructions[:, k]` targets `next_observations[:, K_ctx-1+k]`.
- The reconstruction and foreground reconstruction losses reuse the
  Phase-0 background-subtraction mask helpers from `models.losses`.

## Optimizer

Training entry point: `src/train/train_mechanics.py`. Console script
`train-mechanics`.

Default optimizer settings:

- `torch.optim.AdamW`, `lr = 3e-4`, `weight_decay = 1e-6`
- linear warmup `1000` steps, cosine decay to 0 across the run
- gradient clip `100.0`
- AdamW + cosine schedule + RNG capture/restore + auto-resume + per-step
  validation open-loop probe (`val_open_loop_*` flags) match the Phase-0
  trainer one-for-one so `history.jsonl` is comparable across B1 and the
  mechanics model.

Default loss weights:

```
reconstruction_weight                          = 1.0
foreground_reconstruction_weight               = 1.0
foreground_imagination_reconstruction_weight   = 2.0
imagination_reconstruction_weight              = 0.0
kl_weight                                      = 1.0
kl_balance_alpha                               = 0.8
kl_free_nats_mech                              = 0.5
kl_free_nats_nuisance                          = 3.0
smoothness_weight                              = 1.0
reward_weight, imagined_reward_weight          = 0.0
```

Checkpoints are tagged `model_kind = "mechanics"` so the loader rejects
mismatched (RSSM ↔ mechanics) checkpoint files instead of silently
mis-loading state.

## Diagnostics

`src/eval/probes.py` provides two **non-training** diagnostics:

- `fit_linear_probe`: held-out OLS of `encoded_q -> physics_qpos` (and
  `encoded_qdot -> physics_qvel`). Reports R² per dim and overall.
  R² > 0.8 is the K2 acceptance gate; values below that suggest
  factorization collapse.
- `compute_learned_energy` + `normalized_energy_drift`:
  `E_learned(t) = T(q_t, qdot_t) + V(q_t)` from the model's mass matrix +
  potential. With `D = 0` this should be approximately conserved over open-
  loop rollouts; with `D > 0` it should be monotone-non-increasing on
  average. Used for H3.

## Relation to Other Models

| Model | Code path | What it tests |
| --- | --- | --- |
| B1 RSSM | `models.world_model.VisualWorldModel` | Unstructured baseline |
| **B_main (this)** | `models.mechanics.MechanicsWorldModel` | Full factorization + Lagrangian + dissipation |
| B2 LNN-only | `--no-dissipation` on this model | H3 ablation: how much does `D` help? |
| B3 Unfactored LNN+D | _planned_, share Lagrangian primitives | H2 ablation: does the q/z split matter? |
| B4 Contrastive | `models.contrastive.*` _planned_ | Reconstruction-free baseline |

The model intentionally reuses the Phase-0/1 infrastructure (foreground
mask, imagination rollout API, reward head, CEM planner via
`imagine_rollout_features`, `posterior_step` / `initial_posterior` for
online MPC) so swapping models in `eval-control` is a one-line change.
