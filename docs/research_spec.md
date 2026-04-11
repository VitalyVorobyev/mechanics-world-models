# Mechanics-Structured Visual World Models for Out-of-Distribution Control

*Research Specification — Visual RL on Apple Silicon*

---

## Research question

When do structured mechanical priors in latent world models improve generalization of visual control policies beyond the training regime — specifically under shifted physical parameters (mass, length, friction) and visual nuisance perturbations (camera pose, background, lighting)?

## Hypotheses

**H1 (Physical OOD).** A world model whose latent transition is constrained to Euler–Lagrange dynamics with learned Rayleigh dissipation and actuation will degrade more gracefully under physical parameter shifts than an equivalently-sized RSSM, because the Lagrangian form constrains extrapolation to physically plausible trajectories.

**H2 (Visual OOD).** A factored latent (q, q̇, z_nuisance) trained with a structured transition on (q, q̇) and an independent AR(1) on z will isolate mechanical state from visual style, yielding better control under camera/background shift than an entangled latent of the same total dimension.

**H3 (Dissipation matters).** A pure conservative (energy-preserving) Lagrangian prior will underperform a conservative-plus-dissipative model on any actuated, damped system. The gap will grow with damping coefficient.

## Method

**Architecture.** Conv encoder → factored latent (q ∈ ℝᵈ, q̇ ∈ ℝᵈ, z ∈ ℝᵏ). Transition on (q, q̇): q̈ = (∂²L/∂q̇²)⁻¹ [∂L/∂q − (∂²L/∂q∂q̇)q̇ − ∂D/∂q̇ + B(q)u], where L(q, q̇) is a learned Lagrangian, D(q̇) is a learned Rayleigh dissipation function, B(q) is a learned actuation matrix. Nuisance z follows z′ = αz + ε. Conv decoder reconstructs pixels from (q, q̇, z). Reward head predicts scalar reward from (q, q̇).

**Control.** CEM/MPC on the learned model (H=15 steps, 512 samples, 3 iterations). No actor-critic in primary evaluation — MPC directly tests model quality without confounding policy overfitting. Actor-critic added as secondary comparison for in-distribution throughput.

**Environments.** DeepMind Control Suite with pixel observations (84×84 RGB, action repeat 2): cartpole-swingup (1-DOF), acrobot-swingup (2-DOF), walker-walk (6-DOF, stretch goal). Domain randomization harness for mass, length, friction, camera pose, background.

## Baselines

| Tag | Model | Rationale |
|-----|-------|-----------|
| B1 | RSSM (DreamerV3-small) | Standard unstructured world model |
| B2 | LNN-only (no dissipation) | Ablation: tests H3 (conservative-only) |
| B3 | Unfactored LNN+D (no z split) | Ablation: tests H2 (entangled latent) |
| B4 | Reconstruction-free contrastive | Tests whether decoding hurts OOD robustness |

## Metrics

**Primary:** (1) MPC episode return under physical parameter shift (mass ×0.5, ×1.5, ×2.0; length ×0.5, ×1.5), normalized by nominal return. (2) MPC episode return under visual shift (Distracting Control perturbations: background video, camera jitter, color randomization).

**Secondary:** (3) Multi-step pixel prediction error (10-step, 50-step open-loop MSE). (4) Latent energy tracking error: |E_learned(t) − E_true(t)| over rollout. (5) Sample efficiency: environment steps to 80% of nominal MPC return. (6) Wall-clock training time per 100K steps.

## Kill criteria

**K1 (Week 3).** If the structured model cannot match RSSM in-distribution prediction accuracy (10-step MSE) on cartpole within 2× training time after debugging, the factorization or LNN implementation is broken. **Action:** debug architecture, do not proceed to OOD experiments.

**K2 (Week 5).** If the structured model shows <5% relative improvement over RSSM on *both* physical parameter shift *and* visual nuisance shift on cartpole + acrobot, the hypothesis is wrong for these environments. **Action:** pivot to online system-ID from pixels or visual servoing benchmark.

**K3 (Week 6).** If walker-walk takes >48h per seed, drop it. Report on 1–2 DOF systems only; the paper still works if the generalization story is clean.

## Key prior work (positioned against)

Zhong & Leonard (NeurIPS 2020): Lagrangian from images — prediction only, no OOD evaluation, no RL control. KeyCLD (CoRL 2021): keypoint-based, energy-shaping control, no pixel-to-action, no nuisance shift. DreamSAC (2025): Hamiltonian world model from pixels — unsupervised RL, not evaluated on parameter shift. PIN-WM: physics-informed WM for manipulation — task-specific, not benchmarked on DMC. **This work:** first systematic OOD-generalization evaluation of mechanics-structured visual world models under joint physical + visual distribution shift.

## Compute & timeline

All training on Apple Silicon (M2/M3). Cartpole: 2–6h/run. Acrobot: 4–10h/run. Walker: 12–24h/run. Full experiment grid (2 envs × 5 models × 3 seeds × ~6h avg): ~180 compute-hours ≈ 8 days sequential, parallelizable to ~3 days across cores. Companion Rust data pipeline developed in parallel (weeks 2–6). Target: workshop paper draft at week 8.
