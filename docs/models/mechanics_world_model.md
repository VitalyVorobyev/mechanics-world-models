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

## What The Model Learns

The bet this model is making is that natural dynamical systems have a
lot of structure the unstructured RSSM has to discover from scratch.
Specifically: real mechanical systems are governed by a Lagrangian
`L(q, qdot) = T(q, qdot) − V(q)` with a quadratic-in-qdot kinetic
energy, possibly a Rayleigh dissipation `D(qdot)`, and a linear-in-u
actuation channel `B(q) u`. If the world model is forced to obey this
template then the encoder has a strong incentive to map pixels to
something resembling **generalized coordinates** — position-like
variables `q` whose time derivative is velocity-like `qdot` and whose
evolution is governed by physics.

The latent is split into three pieces, each with a different role:

- **`q` (generalized coordinate, dim `d = 2` on cartpole).** Tracks the
  physically relevant *pose*: for cartpole that is cart position and
  pole angle, though the model is free to use any smooth linear
  reparameterization of these. The Lagrangian step starting from `q_t`
  must predict `q_{t+1}`, and its time-derivative is the model's
  velocity.
- **`qdot` (generalized velocity, dim `d`).** **Derived, not learned
  independently.** Computed in the encoder's forward pass as the
  finite-difference time-derivative of `q`:
  `q̇_t = (q_{t+1} − q_{t−1}) / (2·dt)` in the interior, with forward /
  backward diff at the boundaries. This hard-wires the kinematic
  identity `q̇ = d/dt(q)` by construction; there is no way for the
  encoder to decouple position and velocity, no soft smoothness
  regularizer, no optimization dynamics that can make them drift
  apart. Same pattern as LNN/HNN-from-pixels (Cranmer et al. 2020,
  Greydanus et al. 2019, Toth et al. 2020).
- **`z` (nuisance, dim `k = 4` for cartpole).** Meant to absorb
  everything the mechanics branch does not need to predict the future:
  appearance, lighting, background texture, camera pose, distractor
  patterns. Evolves under a simple AR(1) so it persists but is not
  expected to be forecast from physics. Kept small (`k = 4`) because
  cartpole has essentially no visual style variation; earlier
  iterations with `k = 32` let the encoder abuse `z` for per-frame
  pixel detail, triggering a KL blow-up at step ~300.

Each of the four Lagrangian primitives is small (a two-layer MLP) and
has a specific physical role the model is being pushed to learn:

- **`M(q)`**: the generalized mass / inertia matrix. For a true
  cartpole the exact form involves `m_pole * L * cos(pole_angle)`
  cross-terms — the model is not told this, but a successful `M(q)`
  will exhibit coupling between cart and pole coordinates and will
  vary smoothly with `q`.
- **`V(q)`**: the potential energy landscape. For cartpole this is
  roughly `m * g * L * cos(pole_angle)` (peaked when the pole is
  upright). A trained `V(q)` should show this peak — it is one of the
  easiest sanity checks once a run finishes.
- **`D(qdot)`**: Rayleigh dissipation, PSD by construction. The
  gradient `∂D/∂qdot` is the generalized friction force, monotone in
  `qdot`. For cartpole this should be small and mostly diagonal. If
  the model can match B1's prediction quality with `D = 0` (the
  `--no-dissipation` ablation), H3 is falsified on this environment.
- **`B(q)`**: actuation map. Cartpole has a single motor that pushes
  the cart, so the physically correct `B(q)` is roughly `[1, 0]` (force
  on the cart coordinate, none on the pole). A learned `B(q)` that
  matches this up to the sign and scale of the coordinate
  reparameterization is a confirmation the factorization worked.

Once those four pieces are in place, the forward step is completely
standard physics: solve `M(q) q̈ = ∂L/∂q − Ṁ q̇ − ∂D/∂qdot + B(q) u`
for `q̈`, then advance `(q, qdot)` with a semi-implicit symplectic
step. The nuisance channel updates as `z_{t+1} = α * z_t` with `α`
learnable in `(0, 1)`. The decoder takes `concat(q, qdot, z)` and
produces an image.

Compared to the RSSM, nearly every choice in this model is a
commitment: the mass matrix is PD, the integrator is symplectic, the
KL budget is tighter on the mechanics branch than on the nuisance
branch, and `q̇` is a finite-difference of `q` by construction. Any
one of these commitments could be wrong for a given environment — the
ablations (B2 no-dissipation, B3 no q/z split) exist exactly to probe
this.

## Physical Intuition For The Primitives

A useful way to read a trained checkpoint is to freeze the encoder,
pick a grid of `q` values, and visualize the four primitives. What you
should expect on cartpole once K1 and K2 are met:

| Quantity | Expected shape on cartpole |
| --- | --- |
| `V(q)` | Single maximum near the upright-pole configuration; roughly sinusoidal in the pole-angle coordinate; weak dependence on cart position. |
| `M(q)` | Symmetric PD, with off-diagonal entries that vary smoothly with `q`. Diagonal entries are roughly the effective cart and pole inertias. |
| `D(qdot)` | Small PSD matrix, near-diagonal, gradient roughly linear in `qdot`. Should essentially vanish if the environment is frictionless. |
| `B(q)` | Weight concentrated on the cart coordinate's row; weakly dependent on `q`. |

Conservation checks (via `compute_learned_energy` in
`src/eval/probes.py`) are the other side of the same coin. For a
frictionless rollout `E_learned(t) = T + V` should be nearly constant
over 50+ steps; with the learned `D` engaged it should be monotone
non-increasing on average. This is metric (4) from the research spec
and it is what H3 hinges on.

## Interpreting Inputs And Outputs

**Observations and actions.** Identical contract to the RSSM baseline.
Same encoder trunk.

**Latent factors.** `q`, `qdot`, and `z` live in separate tensors, and
unlike the RSSM's `[h, z]` the individual coordinates are meant to have
real meaning:

- `q_t` can be linearly regressed onto the simulator's `qpos` to
  recover a diagnostic R² (the K2 gate, `fit_linear_probe` in
  `src/eval/probes.py`). Values of R² close to 1 mean the encoder
  found generalized coordinates up to a linear reparameterization.
- `qdot_t` is, by construction, `(q_{t+1} − q_{t−1}) / (2·dt)` (central
  difference interior; forward / backward at boundaries). No loss or
  optimization step is needed to enforce this — it is computed
  explicitly inside `FactoredEncoder.forward`.
- `z_t` is not expected to be interpretable. It should, however, vary
  slowly (`α ≈ 0.95`) and not carry any information the mechanics
  branch needs.

**Reconstructions.** Three modes, parallel to the RSSM:

- *Posterior*: encode each frame, decode `concat(q, qdot, z)`. Expected
  to be sharp almost immediately — the decoder does not depend on the
  dynamics quality.
- *One-step prior*: take the posterior at `t`, run one integrator step,
  decode. Targets `obs_{t+1}`. This is the first place bad dynamics
  show up: cart leaves a trail, pole falls the wrong way, etc.
- *Open-loop imagination*: take `K_ctx` posterior steps to pin down
  `(q, qdot, z)`, then roll the integrator forward `K_imag` steps
  without observations and decode every step. Errors compound. This
  is where we expect structured models to beat the RSSM — the prior
  should stay on the manifold of physically plausible configurations
  for much longer.

**CEM / online MPC.** Because `q̇` is derived by finite differencing,
the encoder requires **two consecutive frames** to produce a velocity;
`initial_posterior` and `posterior_step` accept `[B, T ≥ 2, C, H, W]`
and return `(q, q̇, z)` at the latest timestep. The existing warmup in
`eval-control` (K_ctx ≥ 4) already satisfies this. `imagine_rollout_
features` rolls forward under a candidate action sequence and returns
decoded features suitable for the reward head. The planner does not
know or need to know that the latent is factored — it just sees a
dense feature tensor and a reward head.

## Architecture and Parameters

Implementation entry point:
`src/models/mechanics/world_model.py::MechanicsWorldModel`. Shared notation
is the same as in the RSSM doc; mechanics-specific symbols below.

Default cartpole training configuration:

- observations: `[B, T, 3, 84, 84]`, float32 in `[0, 1]`
- actions: `[B, T, 1]`, float32
- generalized coordinate dim `d = 2` (cartpole)
- nuisance dim `k = 4` (kept small — cartpole has near-zero visual
  style variation, and larger `k` triggered a KL blow-up in earlier
  iterations; see `docs/current_status.md`)
- shared encoder embedding `E = 256`
- decoder input dim `2d + k = 8`
- integrator step `dt = 0.01` s, `n_substeps = 1`

### Encoder (`src/models/mechanics/encoder.py::FactoredEncoder`)

Shared `ConvEncoder` trunk (identical to the RSSM baseline) feeds two heads:

- **MechHead**: `Linear(E, 128) -> ELU -> Linear(128, 2d)` — emits
  `(q_mean, q_raw_std)` per frame. Only `q` is emitted; `q̇` is derived
  from `q` by central / boundary finite differencing inside
  `FactoredEncoder.forward` via `derive_qdot_from_q(q_mean, q_std, dt)`.
- **NuisanceHead**: `Linear(E, 128) -> ELU -> Linear(128, 2k)` — emits
  `(z_mean, z_raw_std)` per frame.

Posterior stds use `softplus(raw) + min_std` with `mech_min_std = 0.05`
and `nuisance_min_std = 1.0` (raised from 0.1 during debugging — see
status doc). The derived `q̇_std` is the Gaussian pushforward of `q_std`
through the diff stencil:
`q̇_std_t = sqrt(q_std_{t+1}² + q_std_{t−1}²) / (2·dt)` in the interior.
Requires `T ≥ 2` along the time axis — at MPC inference, the caller
must buffer two consecutive frames.

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

with three learnable per-coordinate prior stds (`q`, `qdot`, `z`). The
prior std floor for `q̇` defaults to `prior_min_std / dt` so the
integrator's predicted velocity uncertainty matches the scale of the
encoder's pushforward `q̇_std ≈ q_std / dt` — without this auto-scaling,
`KL(posterior q̇ ‖ prior q̇)` explodes at init because of the 100× σ
mismatch under `dt = 0.01`. At `t = 0` the prior is unit Gaussian (no
history to condition on).

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
  + reward_weight * reward_loss
  + imagined_reward_weight * imagined_reward_loss
```

Notable terms specific to this model:

- **Three-branch balanced KL** with a per-branch free-nats floor:
  `kl_free_nats_mech = 0.5` for `q` and `qdot`; `kl_free_nats_nuisance
  = 3.0` for `z`. The asymmetric budget keeps the nuisance branch
  loose so the encoder can park appearance there, while the mechanics
  branch is held tight against the integrator-derived prior. Each
  branch uses DreamerV2-style balancing with `alpha = 0.8`. `kl_qdot`
  is a *consistency check*: the integrator's predicted velocity
  should match the finite-difference velocity derived from `q`. Large
  values mean the learned Lagrangian is inconsistent with the
  observed motion of the encoder's `q`.
- **No smoothness prior.** An earlier iteration of the model had a
  soft smoothness loss (`‖½(q̇_t + q̇_{t+1}) − (q_{t+1} − q_t)/dt‖²`)
  that tied the encoder's two independent mechanics heads into a
  derivative relationship. It proved too weak — reconstruction
  pressure drove `q̇` to encode per-frame pixel detail instead of
  velocity, inflating `q̇` to magnitude ~15 by epoch 6. The current
  design eliminates the problem at the architecture level by deriving
  `q̇` from `q` in the forward pass. No soft constraint needed; none
  possible to drift away from.
- **Imagination rollout** is identical in shape to the Phase-0 RSSM
  imagination contract: a posterior context of `K_ctx` steps followed
  by `K_imag` closed-loop prior steps decoded with the same decoder.
  `imagined_reconstructions[:, k]` targets
  `next_observations[:, K_ctx-1+k]`.
- The reconstruction and foreground reconstruction losses reuse the
  Phase-0 background-subtraction mask helpers from `models.losses`.

## Optimizer

Training entry point: `src/train/train_mechanics.py`. Console script
`train-mechanics`.

Default optimizer settings:

- `torch.optim.AdamW`, `lr = 3e-4`, `weight_decay = 1e-6`
- linear warmup `1000` steps, cosine decay to 0 across the run
- gradient clip `100.0` + a NaN/Inf optimizer-step guard (skip step if
  the loss or clipped gradient norm is non-finite; keeps AdamW's
  running moments intact under cold-start numerical noise)
- AdamW + cosine schedule + RNG capture/restore + auto-resume + per-step
  validation open-loop probe (`val_open_loop_*` flags) match the
  Phase-0 trainer one-for-one so `history.jsonl` is comparable across
  B1 and the mechanics model.

The recommended live-training config (from
`scripts/train_mechanics.sh`, derived by the v2/v3/v4/diag debugging
iterations documented in `docs/current_status.md`):

```
kl_weight                                      = 0.1
kl_balance_alpha                               = 0.8
kl_free_nats_mech                              = 0.5
kl_free_nats_nuisance                          = 3.0
nuisance_min_std                               = 1.0
nuisance_dim                                   = 4
learning_rate                                  = 1e-4
warmup_steps                                   = 2000
foreground_imagination_reconstruction_weight   = 2.0
reconstruction_weight                          = 1.0
foreground_reconstruction_weight               = 1.0
imagination_reconstruction_weight              = 0.0
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

## Signals And Failure Modes

Healthy training looks like the RSSM signals plus a few mechanics-
specific ones:

- **`val_open_loop_fg_mse_h10`** ≤ the B1 value by epoch 6–8 is the K1
  gate for this model. If it stays far above B1 while posterior
  reconstruction is good, the Lagrangian is not fitting the data — the
  structural bias is too strong for this environment or too weak
  (not enough MLP capacity in `M, V, D, B`).
- **`kl_q_raw`**: small (~1 nat) if the Lagrangian's predicted
  position matches the encoder's position observations.
- **`kl_qdot_raw`**: starts moderately high at init (~200 nats) from
  the pushforward mismatch against an untrained integrator; should
  drop as the integrator learns to produce velocities consistent with
  the finite-difference of `q`. Large stable values mean the learned
  Lagrangian is physically inconsistent with the observed motion.
- **`kl_z_raw`**: the nuisance KL. `kl_free_nats_nuisance = 3` means
  the clamped contribution is ≥ 3 nats; raw values in `[5, 30]` are
  the healthy range on cartpole.
- **`z_post_mean_rms`, `z_post_mean_abs_max`, `kl_z_t0_mean`,
  `kl_z_transition_mean`**: diagnostic metrics added after the v3
  debugging. Growing `z_post_mean_rms` or `kl_z_t0_mean` indicates
  posterior escape — z is drifting to unbounded magnitude under the
  weak AR(1) anchor. Currently well-controlled with `nuisance_dim = 4`
  and `nuisance_min_std = 1.0`.
- **`q_post_mean_abs_max`, `qdot_post_mean_abs_max`**: the derived
  `q̇_max` is structurally bounded by `q_max / dt`; if `q` stays
  bounded (it does on cartpole), so does `q̇`.
- **`nuisance_alpha`**: the learned AR(1) coefficient. Should move
  toward 1 as training progresses if `z` is carrying slowly-varying
  appearance, and toward 0 if it is not learning to persist anything.

Typical failure modes unique to this model:

- *Factorization collapse*. All useful information ends up in `z`; `q`
  becomes near-constant. Symptoms: linear probe R² ≈ 0, open-loop
  prediction works but is driven entirely by `z`'s AR(1). Mitigation:
  lower `kl_free_nats_nuisance`, shrink the nuisance dim further.
- *Rigid mechanics*. `q` tracks something reasonable but `M(q)` /
  `V(q)` are nearly constant, so the integrator step degenerates to a
  free-particle update. Symptoms: open-loop at short horizons is
  fine, but long horizons diverge. Mitigation: increase dynamics MLP
  capacity or widen the training horizon.
- *Energy blow-up*. On a frictionless rollout, learned energy grows
  without bound. This is the symptom the symplectic integrator was
  chosen to avoid; if it still happens it usually means `dt` is too
  large for the effective stiffness of the learned `M, V`. Mitigation:
  increase `n_substeps` or shrink `dt`.
- *Dissipation eats everything*. `D` becomes very large early; the
  model uses it to suppress its own predictions and then the
  integrator rolls every state toward the fixed point of the
  potential. Symptoms: everything decays to a canonical pose.
  Mitigation: start `Dissipation.net[-1].bias` near zero (its default
  initialization does this) and/or use `--no-dissipation` as a
  sanity-check ablation.
- *Nuisance posterior escape*. z_mean grows unbounded under the weak
  AR(1) anchor (only the `t=0` term pulls toward zero; see v2/v3
  debugging in the status doc). Mitigation: small `nuisance_dim` (4
  works on cartpole), large `nuisance_min_std` (1.0), watch the
  z-diagnostic metrics for early warning.

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
