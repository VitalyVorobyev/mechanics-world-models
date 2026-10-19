"""Factored mechanics world model: encoder -> (q, qdot, z) -> integrator -> decoder.

Mirrors ``VisualWorldModel`` as closely as possible — same posterior/prior
split, same imagination-rollout contract, same optional reward head — so the
Phase-0 loss plumbing, CEM planner, and eval harness can be dropped in with
minimal surgery. The only structural change is that the transition model is
a learned Lagrangian + AR(1) nuisance instead of a GRU-backed RSSM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from models.decoder import ImageDecoder
from models.mechanics.coordinates import (
    coordinate_feature_size,
    embed_coordinates,
    validate_angle_dims,
    wrap_angle_dims,
)
from models.mechanics.encoder import (
    FactoredEncoder,
    FactoredPosterior,
    KeypointFactoredEncoder,
    derive_qdot_from_q,
)
from models.mechanics.integrator import semi_implicit_euler_step
from models.mechanics.keypoints import GeometricRenderer
from models.mechanics.lagrangian import (
    Actuation,
    Dissipation,
    LagrangianDynamics,
    LinearDissipation,
    MassMatrix,
    MLPDynamics,
    Potential,
)
from models.mechanics.transition import MechanicsPrior, MechanicsTransition


@dataclass(frozen=True)
class MechanicsRollout:
    """All per-timestep tensors produced by the posterior+prior rollout.

    Prior arrays at ``t=0`` hold the initial (unit-ish) prior; at ``t>=1``
    they hold ``transition.step`` outputs conditioned on the posterior-sampled
    state at ``t-1``. Every tensor is ``[B, T, ...]`` so downstream losses can
    index consistently.
    """

    posterior: FactoredPosterior
    prior_q_mean: torch.Tensor
    prior_q_std: torch.Tensor
    prior_qdot_mean: torch.Tensor
    prior_qdot_std: torch.Tensor
    prior_z_mean: torch.Tensor
    prior_z_std: torch.Tensor
    q_sampled: torch.Tensor
    qdot_sampled: torch.Tensor
    z_sampled: torch.Tensor

    @property
    def features_posterior_mean(self) -> torch.Tensor:
        """Decoder input: concat posterior means along the last dim."""

        return torch.cat(
            [self.posterior.q_mean, self.posterior.qdot_mean, self.posterior.z_mean],
            dim=-1,
        )

    @property
    def features_posterior_sample(self) -> torch.Tensor:
        """Sampled posterior features (used for stochastic rollout variants)."""

        return torch.cat([self.q_sampled, self.qdot_sampled, self.z_sampled], dim=-1)


@dataclass(frozen=True)
class MechanicsWorldModelOutput:
    """Top-level outputs for the mechanics model's training loss."""

    rollout: MechanicsRollout
    reconstructions: torch.Tensor
    imagined_reconstructions: torch.Tensor | None = None
    imagined_features: torch.Tensor | None = None
    imagined_q: torch.Tensor | None = None
    imagined_qdot: torch.Tensor | None = None
    imagined_z: torch.Tensor | None = None
    imagined_target_start: int | None = None
    reward_predictions: torch.Tensor | None = None
    imagined_reward_predictions: torch.Tensor | None = None
    # Mean squared (angle-wrapped) error of multi-start latent rollouts against
    # the encoded q of later frames; 0-d, only when latent_dynamics_horizon > 0.
    latent_dynamics_loss: torch.Tensor | None = None
    # Mean squared distance between encoder keypoints and the renderer's
    # keypoints for the same q; 0-d, only with keypoint encoder + renderer.
    keypoint_consistency_loss: torch.Tensor | None = None


NUISANCE_MODES = ("concat", "style")
DYNAMICS_FORMS = ("lagrangian", "mlp")
DISSIPATION_FORMS = ("rayleigh_mlp", "linear")
ENCODER_TYPES = ("conv", "keypoint")
DECODER_TYPES = ("deconv", "render")


def _sample_normal(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(mean)
    return mean + std * noise


class MechanicsWorldModel(nn.Module):
    """Factored ``(q, qdot, z)`` visual world model.

    ``nuisance_mode`` controls how ``z`` reaches the pixels:

    * ``"concat"`` (legacy): the decoder reads ``concat(q, q̇, z)``. Nothing
      stops ``z`` from carrying the pose — on the April 2026 checkpoints a
      probe on ``z`` recovered cart position with R² ≈ 0.85 while ``q`` had
      collapsed (see ``docs/archive/status_history_2026-04_to_2026-10-19.md``).
    * ``"style"``: the decoder renders from the mechanics coordinates only;
      ``z`` enters as a per-channel gain/offset on the decoder logits, which
      can express colour/brightness nuisance but not where the cart is. The
      reward head then reads ``(q, q̇)`` only, as the research spec states.

    ``angle_dims`` marks generalized coordinates living on ``S^1``; the decoder,
    reward head and Lagrangian primitives see them through ``(cos, sin)``.
    """

    def __init__(
        self,
        action_dim: int,
        q_dim: int = 2,
        nuisance_dim: int = 32,
        embedding_size: int = 256,
        encoder_head_hidden_size: int = 128,
        mass_matrix_hidden_size: int = 64,
        potential_hidden_size: int = 64,
        dissipation_hidden_size: int = 64,
        actuation_hidden_size: int = 64,
        dynamics_hidden_layers: int = 2,
        image_size: int = 84,
        input_channels: int = 3,
        dt: float = 0.01,
        n_substeps: int = 1,
        with_dissipation: bool = True,
        mech_min_std: float = 0.05,
        nuisance_min_std: float = 0.1,
        nuisance_alpha_init: float = 0.95,
        reward_head: bool = False,
        reward_head_hidden_size: int = 200,
        qdot_stencil: str = "central",
        angle_dims: Sequence[int] = (),
        nuisance_mode: str = "concat",
        dynamics_form: str = "lagrangian",
        dissipation_form: str = "rayleigh_mlp",
        mlp_dynamics_hidden_size: int = 128,
        encoder_type: str = "conv",
        decoder_type: str = "deconv",
    ) -> None:
        super().__init__()
        if nuisance_mode not in NUISANCE_MODES:
            raise ValueError(f"nuisance_mode must be one of {NUISANCE_MODES}")
        if encoder_type not in ENCODER_TYPES:
            raise ValueError(f"encoder_type must be one of {ENCODER_TYPES}")
        if decoder_type not in DECODER_TYPES:
            raise ValueError(f"decoder_type must be one of {DECODER_TYPES}")
        if (encoder_type == "keypoint" or decoder_type == "render") and (
            q_dim != 2 or tuple(angle_dims) != (1,)
        ):
            raise ValueError(
                "keypoint encoder / render decoder are cartpole-specific: "
                "require q_dim=2 and angle_dims=(1,)",
            )
        if decoder_type == "render" and nuisance_mode != "style":
            raise ValueError("decoder_type='render' requires nuisance_mode='style'")
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.action_dim = action_dim
        self.q_dim = q_dim
        self.nuisance_dim = nuisance_dim
        self.image_size = image_size
        self.input_channels = input_channels
        self.dt = dt
        self.angle_dims = validate_angle_dims(angle_dims, q_dim)
        self.nuisance_mode = nuisance_mode
        coordinate_size = coordinate_feature_size(q_dim, self.angle_dims)
        mechanics_feature_size = coordinate_size + q_dim
        if nuisance_mode == "style":
            decoder_input_size = coordinate_size
            # Reward / imagined features: mechanics state only.
            self.feature_size = mechanics_feature_size
        else:
            decoder_input_size = mechanics_feature_size + nuisance_dim
            self.feature_size = decoder_input_size
        self.reward_head_enabled = reward_head

        self.encoder: FactoredEncoder | KeypointFactoredEncoder
        if encoder_type == "keypoint":
            self.encoder = KeypointFactoredEncoder(
                nuisance_dim=nuisance_dim,
                dt=dt,
                head_hidden_size=encoder_head_hidden_size,
                mech_min_std=mech_min_std,
                nuisance_min_std=nuisance_min_std,
                input_channels=input_channels,
                qdot_stencil=qdot_stencil,
            )
        else:
            self.encoder = FactoredEncoder(
                q_dim=q_dim,
                nuisance_dim=nuisance_dim,
                dt=dt,
                embedding_size=embedding_size,
                head_hidden_size=encoder_head_hidden_size,
                mech_min_std=mech_min_std,
                nuisance_min_std=nuisance_min_std,
                input_channels=input_channels,
                qdot_stencil=qdot_stencil,
                angle_dims=self.angle_dims,
            )
        if dynamics_form not in DYNAMICS_FORMS:
            raise ValueError(f"dynamics_form must be one of {DYNAMICS_FORMS}")
        if dissipation_form not in DISSIPATION_FORMS:
            raise ValueError(f"dissipation_form must be one of {DISSIPATION_FORMS}")
        self.dynamics_form = dynamics_form
        self.dynamics: LagrangianDynamics | MLPDynamics
        if dynamics_form == "mlp":
            self.dynamics = MLPDynamics(
                q_dim=q_dim,
                action_dim=action_dim,
                hidden_size=mlp_dynamics_hidden_size,
                hidden_layers=dynamics_hidden_layers,
                angle_dims=self.angle_dims,
            )
        else:
            dissipation: Dissipation | LinearDissipation | None = None
            if with_dissipation and dissipation_form == "linear":
                dissipation = LinearDissipation(q_dim=q_dim)
            elif with_dissipation:
                dissipation = Dissipation(
                    q_dim=q_dim,
                    hidden_size=dissipation_hidden_size,
                    hidden_layers=dynamics_hidden_layers,
                )
            self.dynamics = LagrangianDynamics(
                mass_matrix=MassMatrix(
                    q_dim=q_dim,
                    hidden_size=mass_matrix_hidden_size,
                    hidden_layers=dynamics_hidden_layers,
                    angle_dims=self.angle_dims,
                ),
                potential=Potential(
                    q_dim=q_dim,
                    hidden_size=potential_hidden_size,
                    hidden_layers=dynamics_hidden_layers,
                    angle_dims=self.angle_dims,
                ),
                dissipation=dissipation,
                actuation=Actuation(
                    q_dim=q_dim,
                    action_dim=action_dim,
                    hidden_size=actuation_hidden_size,
                    hidden_layers=dynamics_hidden_layers,
                    angle_dims=self.angle_dims,
                ),
            )
        self.transition = MechanicsTransition(
            dynamics=self.dynamics,
            nuisance_dim=nuisance_dim,
            dt=dt,
            n_substeps=n_substeps,
            prior_min_std=mech_min_std,
            nuisance_prior_min_std=nuisance_min_std,
            nuisance_alpha_init=nuisance_alpha_init,
        )
        self.decoder: ImageDecoder | GeometricRenderer
        if decoder_type == "render":
            self.decoder = GeometricRenderer(output_channels=input_channels, output_size=image_size)
        else:
            self.decoder = ImageDecoder(
                feature_size=decoder_input_size,
                output_channels=input_channels,
                output_size=image_size,
            )
        self.style_head: nn.Linear | None = None
        if nuisance_mode == "style":
            # Per-channel (gain, offset) on the decoder logits; zero-init so
            # training starts from the unmodulated mechanics-only render.
            self.style_head = nn.Linear(nuisance_dim, 2 * input_channels)
            nn.init.zeros_(self.style_head.weight)
            nn.init.zeros_(self.style_head.bias)
        # Register dynamics submodules so their parameters are tracked; the
        # LagrangianDynamics dataclass holds them but isn't an nn.Module.
        self._register_dynamics_modules()

        self.reward_head: nn.Module | None
        if reward_head:
            self.reward_head = nn.Sequential(
                nn.Linear(self.feature_size, reward_head_hidden_size),
                nn.ELU(),
                nn.Linear(reward_head_hidden_size, reward_head_hidden_size),
                nn.ELU(),
                nn.Linear(reward_head_hidden_size, 1),
            )
        else:
            self.reward_head = None

    def _register_dynamics_modules(self) -> None:
        """Expose the dynamics primitives to nn.Module so state_dict captures them.

        Transition already holds its learnable std parameters, but the
        Lagrangian primitives live inside a dataclass — we need to add them
        as named submodules so ``.parameters()`` and checkpoint save/load
        behave correctly.
        """

        if isinstance(self.dynamics, MLPDynamics):
            return  # an nn.Module itself, already registered as ``self.dynamics``
        self.mass_matrix_module = self.dynamics.mass_matrix
        self.potential_module = self.dynamics.potential
        self.actuation_module = self.dynamics.actuation
        if self.dynamics.dissipation is not None:
            self.dissipation_module = self.dynamics.dissipation

    def reward_features(
        self, q: torch.Tensor, qdot: torch.Tensor, z: torch.Tensor,
    ) -> torch.Tensor:
        """Reward-head / planner features: ``[..., feature_size]``.

        ``concat`` mode keeps the legacy ``concat(q, q̇, z)`` (with angle dims
        embedded); ``style`` mode drops ``z``.
        """

        mechanics = torch.cat([embed_coordinates(q, self.angle_dims), qdot], dim=-1)
        if self.nuisance_mode == "style":
            return mechanics
        return torch.cat([mechanics, z], dim=-1)

    def decode(
        self, q: torch.Tensor, qdot: torch.Tensor, z: torch.Tensor,
    ) -> torch.Tensor:
        """Render ``[B, T, C, H, W]`` images from ``[B, T, ·]`` latent means/samples."""

        if self.nuisance_mode == "style":
            assert self.style_head is not None
            if isinstance(self.decoder, GeometricRenderer):
                flat = self.decoder(q.reshape(-1, q.shape[-1]))
                logits = flat.reshape(*q.shape[:-1], *flat.shape[1:])
            else:
                logits = self.decoder.decode_logits(embed_coordinates(q, self.angle_dims))
            gain, offset = torch.chunk(self.style_head(z), chunks=2, dim=-1)  # [B, T, C]
            logits = logits * (1.0 + gain[..., None, None]) + offset[..., None, None]
            return torch.sigmoid(logits)
        return self.decoder(self.reward_features(q, qdot, z))

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        imagination_context_steps: int = 0,
        imagination_horizon: int = 0,
        latent_dynamics_horizon: int = 0,
    ) -> MechanicsWorldModelOutput:
        """Run a posterior+prior pass with optional prior imagination.

        ``latent_dynamics_horizon > 0`` also rolls the dynamics that many
        steps from every frame and scores it against the encoded ``q``
        (KeyCLD-style multi-start loss), which gives dense dynamics gradients
        without decoding.
        """

        posterior = self.encoder(observations)
        rollout = self._rollout(posterior=posterior, actions=actions)
        reconstructions = self.decode(posterior.q_mean, posterior.qdot_mean, posterior.z_mean)
        reward_predictions: torch.Tensor | None = None
        if self.reward_head is not None:
            features = self.reward_features(
                posterior.q_mean, posterior.qdot_mean, posterior.z_mean,
            )
            reward_predictions = self.reward_head(features).squeeze(-1)

        imagined_reconstructions: torch.Tensor | None = None
        imagined_features: torch.Tensor | None = None
        imagined_q: torch.Tensor | None = None
        imagined_qdot: torch.Tensor | None = None
        imagined_z: torch.Tensor | None = None
        imagined_target_start: int | None = None
        imagined_reward_predictions: torch.Tensor | None = None
        if imagination_horizon > 0 or imagination_context_steps > 0:
            (
                imagined_reconstructions,
                imagined_features,
                imagined_q,
                imagined_qdot,
                imagined_z,
                imagined_target_start,
            ) = self.imagine_from_posterior(
                rollout=rollout,
                actions=actions,
                context_steps=imagination_context_steps,
                horizon=imagination_horizon,
            )
            if self.reward_head is not None and imagined_features is not None:
                imagined_reward_predictions = self.reward_head(imagined_features).squeeze(-1)

        latent_dynamics_loss: torch.Tensor | None = None
        if latent_dynamics_horizon > 0:
            latent_dynamics_loss = self.multi_start_latent_loss(
                posterior, actions, latent_dynamics_horizon,
            )
        keypoint_consistency_loss: torch.Tensor | None = None
        if posterior.keypoints is not None and isinstance(self.decoder, GeometricRenderer):
            rendered = self.decoder.keypoints(posterior.q_mean)
            keypoint_consistency_loss = (posterior.keypoints - rendered).square().sum(-1).mean()

        return MechanicsWorldModelOutput(
            latent_dynamics_loss=latent_dynamics_loss,
            keypoint_consistency_loss=keypoint_consistency_loss,
            rollout=rollout,
            reconstructions=reconstructions,
            imagined_reconstructions=imagined_reconstructions,
            imagined_features=imagined_features,
            imagined_q=imagined_q,
            imagined_qdot=imagined_qdot,
            imagined_z=imagined_z,
            imagined_target_start=imagined_target_start,
            reward_predictions=reward_predictions,
            imagined_reward_predictions=imagined_reward_predictions,
        )

    def multi_start_latent_loss(
        self,
        posterior: FactoredPosterior,
        actions: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Roll ``horizon`` integrator steps from every usable frame; L2 vs encoded q.

        Starts at ``t = 1 .. T - 1 - horizon`` (``t = 0`` is skipped because
        its finite-difference ``q̇`` reads frame 1). Start states use the
        posterior means; targets are the encoder's ``q_{t+k}``, detached so
        the loss trains the dynamics rather than pulling the encoder towards
        whatever the dynamics predict. Angle residuals are wrapped.
        Shapes: posterior ``[B, T, d]``, actions ``[B, T, A]``.
        """

        batch, steps, q_dim = posterior.q_mean.shape
        num_starts = steps - 1 - horizon
        if num_starts < 1:
            raise ValueError(
                f"latent_dynamics_horizon={horizon} needs sequence_length >= {horizon + 2}",
            )
        q = posterior.q_mean[:, 1 : 1 + num_starts].reshape(-1, q_dim)
        qdot = posterior.qdot_mean[:, 1 : 1 + num_starts].reshape(-1, q_dim)
        targets = posterior.q_mean.detach()
        squared = q.new_zeros(())
        for k in range(horizon):
            action = actions[:, 1 + k : 1 + k + num_starts].reshape(-1, actions.shape[-1])
            q, qdot = semi_implicit_euler_step(
                dynamics=self.dynamics,
                q=q,
                qdot=qdot,
                u=action,
                dt=self.dt,
                n_substeps=self.transition.n_substeps,
            )
            target = targets[:, 2 + k : 2 + k + num_starts].reshape(-1, q_dim)
            residual = wrap_angle_dims(q - target, self.angle_dims)
            squared = squared + residual.square().mean()
        return squared / horizon

    def _rollout(
        self,
        posterior: FactoredPosterior,
        actions: torch.Tensor,
    ) -> MechanicsRollout:
        """Run the posterior rollout, sampling at each step and running priors.

        The prior at step ``t`` depends on the posterior sample at ``t-1`` and
        ``action_{t-1}`` only, so the ``T-1`` integrator calls for ``t=1..T-1``
        are mutually independent. We exploit this by drawing all posterior
        samples up front and then calling ``self.transition.step`` once on a
        flattened batch of size ``B * (T-1)`` instead of looping ``T-1`` times
        in Python — same outputs and gradients, dramatically fewer kernel
        launches on the dynamics path. Imagination cannot be batched this way
        (each imagined step truly depends on the previous step's state) so the
        sequential loop lives only in ``imagine_from_posterior``.
        """

        batch_size, sequence_length = posterior.q_mean.shape[:2]
        device = posterior.q_mean.device
        dtype = posterior.q_mean.dtype

        # Reparameterized samples drawn once across the whole sequence; the
        # randomness pattern differs from the prior-step-then-sample loop, but
        # only the order of consumed RNG draws changes — distributionally each
        # sample is identical to before.
        q_samples = _sample_normal(posterior.q_mean, posterior.q_std)
        # q̇ samples are the finite difference of the q samples (not an
        # independent draw), so the kinematic constraint holds per sample and
        # not only for the posterior means.
        qdot_samples, _ = derive_qdot_from_q(
            q_samples,
            posterior.q_std,
            dt=self.dt,
            stencil=self.encoder.qdot_stencil,
            angle_dims=self.angle_dims,
        )
        z_samples = _sample_normal(posterior.z_mean, posterior.z_std)

        # ``initial`` (t=0) is a fixed unit-Gaussian prior with no history.
        initial = self.transition.initial_prior(batch_size, device=device, dtype=dtype)

        if sequence_length == 1:
            return MechanicsRollout(
                posterior=posterior,
                prior_q_mean=initial.q_mean.unsqueeze(1),
                prior_q_std=initial.q_std.unsqueeze(1),
                prior_qdot_mean=initial.qdot_mean.unsqueeze(1),
                prior_qdot_std=initial.qdot_std.unsqueeze(1),
                prior_z_mean=initial.z_mean.unsqueeze(1),
                prior_z_std=initial.z_std.unsqueeze(1),
                q_sampled=q_samples,
                qdot_sampled=qdot_samples,
                z_sampled=z_samples,
            )

        # Batched integrator for t=1..T-1. Each (b, t) entry is independent.
        flat_batch = batch_size * (sequence_length - 1)
        prev_q_flat = q_samples[:, :-1].reshape(flat_batch, q_samples.shape[-1])
        prev_qdot_flat = qdot_samples[:, :-1].reshape(flat_batch, qdot_samples.shape[-1])
        prev_z_flat = z_samples[:, :-1].reshape(flat_batch, z_samples.shape[-1])
        actions_flat = actions[:, :-1].reshape(flat_batch, actions.shape[-1])

        prior_flat = self.transition.step(
            q_prev=prev_q_flat,
            qdot_prev=prev_qdot_flat,
            z_prev=prev_z_flat,
            action=actions_flat,
        )

        def _unflatten(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch_size, sequence_length - 1, tensor.shape[-1])

        prior_q_mean = torch.cat(
            [initial.q_mean.unsqueeze(1), _unflatten(prior_flat.q_mean)], dim=1,
        )
        if self.angle_dims:
            # Represent each prior angle by the 2*pi-equivalent value nearest
            # the posterior mean, so the KL measures angular distance rather
            # than penalizing an integrator step across the +-pi branch cut.
            prior_q_mean = posterior.q_mean + wrap_angle_dims(
                prior_q_mean - posterior.q_mean, self.angle_dims,
            )
        prior_q_std = torch.cat(
            [initial.q_std.unsqueeze(1), _unflatten(prior_flat.q_std)], dim=1,
        )
        prior_qdot_mean = torch.cat(
            [initial.qdot_mean.unsqueeze(1), _unflatten(prior_flat.qdot_mean)], dim=1,
        )
        prior_qdot_std = torch.cat(
            [initial.qdot_std.unsqueeze(1), _unflatten(prior_flat.qdot_std)], dim=1,
        )
        prior_z_mean = torch.cat(
            [initial.z_mean.unsqueeze(1), _unflatten(prior_flat.z_mean)], dim=1,
        )
        prior_z_std = torch.cat(
            [initial.z_std.unsqueeze(1), _unflatten(prior_flat.z_std)], dim=1,
        )

        return MechanicsRollout(
            posterior=posterior,
            prior_q_mean=prior_q_mean,
            prior_q_std=prior_q_std,
            prior_qdot_mean=prior_qdot_mean,
            prior_qdot_std=prior_qdot_std,
            prior_z_mean=prior_z_mean,
            prior_z_std=prior_z_std,
            q_sampled=q_samples,
            qdot_sampled=qdot_samples,
            z_sampled=z_samples,
        )

    def imagine_from_posterior(
        self,
        rollout: MechanicsRollout,
        actions: torch.Tensor,
        context_steps: int,
        horizon: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """Close the loop after ``context_steps`` and roll the prior ``horizon`` steps.

        Mirrors ``VisualWorldModel.imagine_from_posterior``: the first imagined
        target is ``obs_{K_ctx}`` aligned at ``next_observations[:, K_ctx - 1]``
        for the transition convention used by the loss.
        """

        if context_steps < 1:
            raise ValueError("imagination_context_steps must be >= 1")
        if horizon < 1:
            raise ValueError("imagination_horizon must be >= 1")
        sequence_length = actions.shape[1]
        if sequence_length < context_steps + horizon - 1:
            raise ValueError(
                "imagination requires sequence_length >= context_steps + horizon - 1; "
                f"got T={sequence_length}, K_ctx={context_steps}, K_imag={horizon}",
            )

        q = rollout.posterior.q_mean[:, context_steps - 1]
        if context_steps >= 2:
            # Causal start velocity: a backward difference over the last two
            # context frames. The encoder's central-difference q̇ at
            # ``context_steps - 1`` would read ``q`` of frame ``context_steps``,
            # which is the first open-loop target.
            qdot = wrap_angle_dims(
                q - rollout.posterior.q_mean[:, context_steps - 2], self.angle_dims,
            ) / self.dt
        else:
            qdot = rollout.posterior.qdot_mean[:, context_steps - 1]
        z = rollout.posterior.z_mean[:, context_steps - 1]
        rollout_actions = actions[:, context_steps - 1 : context_steps - 1 + horizon]
        alpha = self.transition.nuisance_alpha

        q_list: list[torch.Tensor] = []
        qdot_list: list[torch.Tensor] = []
        z_list: list[torch.Tensor] = []
        for step in range(horizon):
            action = rollout_actions[:, step]
            q, qdot = semi_implicit_euler_step(
                dynamics=self.dynamics,
                q=q,
                qdot=qdot,
                u=action,
                dt=self.dt,
                n_substeps=self.transition.n_substeps,
            )
            z = alpha * z
            q_list.append(q)
            qdot_list.append(qdot)
            z_list.append(z)
        imagined_q = torch.stack(q_list, dim=1)
        imagined_qdot = torch.stack(qdot_list, dim=1)
        imagined_z = torch.stack(z_list, dim=1)
        imagined_features = self.reward_features(imagined_q, imagined_qdot, imagined_z)
        imagined_reconstructions = self.decode(imagined_q, imagined_qdot, imagined_z)
        return (
            imagined_reconstructions,
            imagined_features,
            imagined_q,
            imagined_qdot,
            imagined_z,
            context_steps - 1,
        )

    def imagine_rollout_features(
        self,
        q_start: torch.Tensor,
        qdot_start: torch.Tensor,
        z_start: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Batched primitive for the CEM planner to score candidate actions."""

        q = q_start
        qdot = qdot_start
        z = z_start
        alpha = self.transition.nuisance_alpha
        horizon = actions.shape[1]
        features_list: list[torch.Tensor] = []
        for step in range(horizon):
            q, qdot = semi_implicit_euler_step(
                dynamics=self.dynamics,
                q=q,
                qdot=qdot,
                u=actions[:, step],
                dt=self.dt,
                n_substeps=self.transition.n_substeps,
            )
            z = alpha * z
            features_list.append(self.reward_features(q, qdot, z))
        return torch.stack(features_list, dim=1)

    def initial_posterior(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(q_last, qdot_last, z_last)`` posterior means from a buffer.

        Requires ``T >= 2`` consecutive frames — the encoder derives ``q̇`` by
        finite differencing, so a single frame is not enough. We return the
        posterior at the latest frame (t = T-1), since that's the step the MPC
        loop is about to act on.

        Args:
            observations: ``[B, T, C, H, W]`` or ``[T, C, H, W]``. ``T >= 2``.

        Returns:
            ``(q, qdot, z)`` posterior means at the last timestep.
        """

        if observations.ndim == 4:
            observations = observations.unsqueeze(0)
        if observations.ndim != 5 or observations.shape[2] != self.input_channels:
            raise ValueError(
                f"observations must have shape [B, T, C={self.input_channels}, H, W] "
                f"or [T, C, H, W]; got {tuple(observations.shape)}",
            )
        if observations.shape[1] < 2:
            raise ValueError(
                "initial_posterior requires T >= 2 frames so qdot can be "
                f"derived by finite differencing; got T={observations.shape[1]}",
            )
        posterior = self.encoder(observations)
        return (
            posterior.q_mean[:, -1],
            posterior.qdot_mean[:, -1],
            posterior.z_mean[:, -1],
        )

    def posterior_step(
        self,
        q_prev: torch.Tensor,
        qdot_prev: torch.Tensor,
        z_prev: torch.Tensor,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Online MPC helper: refresh the posterior from a ``(prev, new)`` pair.

        Unlike the per-frame encoder we had before the finite-diff constraint,
        the encoder now needs two consecutive frames to produce a velocity.
        ``observations`` must therefore be ``[B, 2, C, H, W]`` — the caller
        (CEM/MPC loop) is responsible for buffering the previous frame
        alongside the new one.

        ``q_prev``, ``qdot_prev``, ``z_prev`` are accepted for API parity with
        ``VisualWorldModel.posterior_step`` but ignored — the encoder is
        deterministic given the two buffered frames, so previous state carries
        no extra information here.
        """

        del q_prev, qdot_prev, z_prev  # unused; encoder state lives in the obs buffer
        return self.initial_posterior(observations)
