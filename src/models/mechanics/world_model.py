"""Factored mechanics world model: encoder -> (q, qdot, z) -> integrator -> decoder.

Mirrors ``VisualWorldModel`` as closely as possible — same posterior/prior
split, same imagination-rollout contract, same optional reward head — so the
Phase-0 loss plumbing, CEM planner, and eval harness can be dropped in with
minimal surgery. The only structural change is that the transition model is
a learned Lagrangian + AR(1) nuisance instead of a GRU-backed RSSM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.decoder import ImageDecoder
from models.mechanics.encoder import FactoredEncoder, FactoredPosterior
from models.mechanics.integrator import semi_implicit_euler_step
from models.mechanics.lagrangian import (
    Actuation,
    Dissipation,
    LagrangianDynamics,
    MassMatrix,
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


def _sample_normal(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(mean)
    return mean + std * noise


class MechanicsWorldModel(nn.Module):
    """Factored ``(q, qdot, z)`` visual world model."""

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
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.q_dim = q_dim
        self.nuisance_dim = nuisance_dim
        self.image_size = image_size
        self.input_channels = input_channels
        self.dt = dt
        self.feature_size = 2 * q_dim + nuisance_dim
        self.reward_head_enabled = reward_head

        self.encoder = FactoredEncoder(
            q_dim=q_dim,
            nuisance_dim=nuisance_dim,
            embedding_size=embedding_size,
            head_hidden_size=encoder_head_hidden_size,
            mech_min_std=mech_min_std,
            nuisance_min_std=nuisance_min_std,
            input_channels=input_channels,
        )
        self.dynamics = LagrangianDynamics(
            mass_matrix=MassMatrix(
                q_dim=q_dim,
                hidden_size=mass_matrix_hidden_size,
                hidden_layers=dynamics_hidden_layers,
            ),
            potential=Potential(
                q_dim=q_dim,
                hidden_size=potential_hidden_size,
                hidden_layers=dynamics_hidden_layers,
            ),
            dissipation=(
                Dissipation(
                    q_dim=q_dim,
                    hidden_size=dissipation_hidden_size,
                    hidden_layers=dynamics_hidden_layers,
                )
                if with_dissipation
                else None
            ),
            actuation=Actuation(
                q_dim=q_dim,
                action_dim=action_dim,
                hidden_size=actuation_hidden_size,
                hidden_layers=dynamics_hidden_layers,
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
        self.decoder = ImageDecoder(
            feature_size=self.feature_size,
            output_channels=input_channels,
            output_size=image_size,
        )
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

        self.mass_matrix_module = self.dynamics.mass_matrix
        self.potential_module = self.dynamics.potential
        self.actuation_module = self.dynamics.actuation
        if self.dynamics.dissipation is not None:
            self.dissipation_module = self.dynamics.dissipation

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        imagination_context_steps: int = 0,
        imagination_horizon: int = 0,
    ) -> MechanicsWorldModelOutput:
        """Run a posterior+prior pass with optional prior imagination."""

        posterior = self.encoder(observations)
        rollout = self._rollout(posterior=posterior, actions=actions)
        features = rollout.features_posterior_mean
        reconstructions = self.decoder(features)
        reward_predictions: torch.Tensor | None = None
        if self.reward_head is not None:
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

        return MechanicsWorldModelOutput(
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

    def _rollout(
        self,
        posterior: FactoredPosterior,
        actions: torch.Tensor,
    ) -> MechanicsRollout:
        """Run the posterior rollout, sampling at each step and running priors.

        Priors are conditioned on the posterior *sample* at ``t-1``; this keeps
        the KL term reparameterized and matches standard RSSM practice.
        """

        batch_size, sequence_length = posterior.q_mean.shape[:2]
        device = posterior.q_mean.device
        dtype = posterior.q_mean.dtype

        prior_q_means: list[torch.Tensor] = []
        prior_q_stds: list[torch.Tensor] = []
        prior_qdot_means: list[torch.Tensor] = []
        prior_qdot_stds: list[torch.Tensor] = []
        prior_z_means: list[torch.Tensor] = []
        prior_z_stds: list[torch.Tensor] = []
        q_samples: list[torch.Tensor] = []
        qdot_samples: list[torch.Tensor] = []
        z_samples: list[torch.Tensor] = []

        initial = self.transition.initial_prior(batch_size, device=device, dtype=dtype)
        prev_q: torch.Tensor | None = None
        prev_qdot: torch.Tensor | None = None
        prev_z: torch.Tensor | None = None

        for step in range(sequence_length):
            if step == 0:
                prior = initial
            else:
                assert prev_q is not None and prev_qdot is not None and prev_z is not None
                prior = self.transition.step(
                    q_prev=prev_q,
                    qdot_prev=prev_qdot,
                    z_prev=prev_z,
                    action=actions[:, step - 1],
                )
            prior_q_means.append(prior.q_mean)
            prior_q_stds.append(prior.q_std)
            prior_qdot_means.append(prior.qdot_mean)
            prior_qdot_stds.append(prior.qdot_std)
            prior_z_means.append(prior.z_mean)
            prior_z_stds.append(prior.z_std)

            q_t = _sample_normal(posterior.q_mean[:, step], posterior.q_std[:, step])
            qdot_t = _sample_normal(posterior.qdot_mean[:, step], posterior.qdot_std[:, step])
            z_t = _sample_normal(posterior.z_mean[:, step], posterior.z_std[:, step])
            q_samples.append(q_t)
            qdot_samples.append(qdot_t)
            z_samples.append(z_t)

            prev_q, prev_qdot, prev_z = q_t, qdot_t, z_t

        return MechanicsRollout(
            posterior=posterior,
            prior_q_mean=torch.stack(prior_q_means, dim=1),
            prior_q_std=torch.stack(prior_q_stds, dim=1),
            prior_qdot_mean=torch.stack(prior_qdot_means, dim=1),
            prior_qdot_std=torch.stack(prior_qdot_stds, dim=1),
            prior_z_mean=torch.stack(prior_z_means, dim=1),
            prior_z_std=torch.stack(prior_z_stds, dim=1),
            q_sampled=torch.stack(q_samples, dim=1),
            qdot_sampled=torch.stack(qdot_samples, dim=1),
            z_sampled=torch.stack(z_samples, dim=1),
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
        imagined_features = torch.cat([imagined_q, imagined_qdot, imagined_z], dim=-1)
        imagined_reconstructions = self.decoder(imagined_features)
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
            features_list.append(torch.cat([q, qdot, z], dim=-1))
        return torch.stack(features_list, dim=1)

    def initial_posterior(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(q_0, qdot_0, z_0)`` posterior means for a single frame.

        Used by the online MPC loop. We return means (not samples) so
        replanning at the same timestep is deterministic.
        """

        if observation.ndim == 3:
            observation = observation.unsqueeze(0)
        if observation.ndim != 4 or observation.shape[1] != self.input_channels:
            raise ValueError(
                f"observation must have shape [B, C={self.input_channels}, H, W]; "
                f"got {tuple(observation.shape)}",
            )
        posterior = self.encoder(observation.unsqueeze(1))
        return (
            posterior.q_mean.squeeze(1),
            posterior.qdot_mean.squeeze(1),
            posterior.z_mean.squeeze(1),
        )

    def posterior_step(
        self,
        q_prev: torch.Tensor,
        qdot_prev: torch.Tensor,
        z_prev: torch.Tensor,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Online MPC helper: feed the next observation to refresh the posterior.

        Unlike RSSM, the factored encoder is per-frame and independent of the
        previous latent, so we just re-encode the new frame. ``q_prev`` and
        friends are accepted and ignored so the call signature stays
        parallel to ``VisualWorldModel.posterior_step``.
        """

        del q_prev, qdot_prev, z_prev  # unused; encoder is memoryless per frame
        return self.initial_posterior(observation)
