"""Top-level visual world model wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.decoder import ImageDecoder
from models.encoder import ConvEncoder
from models.rssm import RSSM, RSSMOutput


@dataclass(frozen=True)
class WorldModelOutput:
    """Outputs needed by the world-model loss."""

    embeddings: torch.Tensor
    rssm: RSSMOutput
    reconstructions: torch.Tensor
    next_embeddings: torch.Tensor | None = None
    predicted_next_embeddings: torch.Tensor | None = None
    next_prior_features: torch.Tensor | None = None
    next_reconstructions: torch.Tensor | None = None
    imagined_reconstructions: torch.Tensor | None = None
    imagined_features: torch.Tensor | None = None
    imagined_target_start: int | None = None
    reward_predictions: torch.Tensor | None = None
    imagined_reward_predictions: torch.Tensor | None = None


class VisualWorldModel(nn.Module):
    """Encode observations, infer RSSM states, decode, and predict next embeddings."""

    def __init__(
        self,
        action_dim: int,
        embedding_size: int,
        hidden_size: int,
        latent_size: int,
        image_size: int = 84,
        input_channels: int = 3,
        reward_head: bool = False,
        reward_head_hidden_size: int = 200,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.image_size = image_size
        self.input_channels = input_channels
        self.reward_head_enabled = reward_head

        self.encoder = ConvEncoder(embedding_size=embedding_size, input_channels=input_channels)
        self.rssm = RSSM(
            action_dim=action_dim,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            latent_size=latent_size,
        )
        self.decoder = ImageDecoder(
            feature_size=hidden_size + latent_size,
            output_channels=input_channels,
            output_size=image_size,
        )
        self.latent_predictor = nn.Sequential(
            nn.Linear(hidden_size + latent_size, embedding_size),
            nn.ELU(),
            nn.Linear(embedding_size, embedding_size),
        )
        # Reward head is opt-in: MPC and the research-spec primary metric need
        # it, but the Phase-0 prediction-only baseline does not. Adding this
        # flag preserves checkpoint compatibility — old checkpoints have no
        # reward_head.* keys, and load_checkpoint treats those as new optional
        # parameters initialized from defaults.
        self.reward_head: nn.Module | None
        if reward_head:
            self.reward_head = nn.Sequential(
                nn.Linear(hidden_size + latent_size, reward_head_hidden_size),
                nn.ELU(),
                nn.Linear(reward_head_hidden_size, reward_head_hidden_size),
                nn.ELU(),
                nn.Linear(reward_head_hidden_size, 1),
            )
        else:
            self.reward_head = None

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor | None = None,
        imagination_context_steps: int = 0,
        imagination_horizon: int = 0,
    ) -> WorldModelOutput:
        """Run a posterior world-model pass with optional prior imagination.

        Args:
            observations: Images shaped ``[B, T, 3, 84, 84]``.
            actions: Actions shaped ``[B, T, A]``.
            next_observations: Optional next images shaped ``[B, T, 3, 84, 84]``.
            imagination_context_steps: Number of posterior warmup steps
                ``K_ctx``. The prior is rolled forward from the posterior state
                at index ``K_ctx - 1``. Set to 0 to disable imagination.
            imagination_horizon: Number of closed-loop prior steps ``K_imag``
                to decode. Set to 0 to disable imagination. When enabled, the
                training sequence must satisfy ``T >= K_ctx + K_imag - 1`` so
                every imagined step has an aligned action and target.

        Returns:
            Embeddings ``[B, T, E]``, RSSM tensors, reconstructions
            ``[B, T, 3, 84, 84]``, and optional transition-aligned /
            imagined tensors.
        """

        embeddings = self.encoder(observations)
        rssm_output = self.rssm(embeddings=embeddings, actions=actions)
        # Decode from the posterior mean (not the reparameterized sample) so the
        # decoder is trained on the exact representation used at eval time. The
        # sampled z still lives inside the RSSM rollout, advancing h_{t+1} and
        # shaping the KL term, which is what carries the stochasticity signal.
        posterior_features = rssm_output.features_posterior_mean
        reconstructions = self.decoder(posterior_features)
        reward_predictions: torch.Tensor | None = None
        if self.reward_head is not None:
            reward_predictions = self.reward_head(posterior_features).squeeze(-1)
        next_embeddings: torch.Tensor | None = None
        predicted_next_embeddings: torch.Tensor | None = None
        next_prior_features: torch.Tensor | None = None
        next_reconstructions: torch.Tensor | None = None
        if next_observations is not None:
            # Transition-aligned path: obs_t/action_t predicts the embedding of
            # obs_{t+1}. Features are [B, T, H + Z], embeddings are [B, T, E],
            # and decoded transition predictions are [B, T, 3, 84, 84].
            next_embeddings = self.encoder(next_observations).detach()
            next_prior_features = self.transition_aligned_next_prior_features(embeddings, actions)
            predicted_next_embeddings = self.latent_predictor(next_prior_features)
            next_reconstructions = self.decoder(next_prior_features)

        imagined_reconstructions: torch.Tensor | None = None
        imagined_features: torch.Tensor | None = None
        imagined_target_start: int | None = None
        imagined_reward_predictions: torch.Tensor | None = None
        if imagination_horizon > 0 or imagination_context_steps > 0:
            imagined_reconstructions, imagined_features, imagined_target_start = (
                self.imagine_from_posterior(
                    rssm_output=rssm_output,
                    actions=actions,
                    context_steps=imagination_context_steps,
                    horizon=imagination_horizon,
                )
            )
            if self.reward_head is not None and imagined_features is not None:
                imagined_reward_predictions = self.reward_head(imagined_features).squeeze(-1)

        return WorldModelOutput(
            embeddings=embeddings,
            rssm=rssm_output,
            reconstructions=reconstructions,
            next_embeddings=next_embeddings,
            predicted_next_embeddings=predicted_next_embeddings,
            next_prior_features=next_prior_features,
            next_reconstructions=next_reconstructions,
            imagined_reconstructions=imagined_reconstructions,
            imagined_features=imagined_features,
            imagined_target_start=imagined_target_start,
            reward_predictions=reward_predictions,
            imagined_reward_predictions=imagined_reward_predictions,
        )

    def imagine_from_posterior(
        self,
        rssm_output: RSSMOutput,
        actions: torch.Tensor,
        context_steps: int,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Close the loop after ``context_steps`` and roll the prior ``horizon`` steps.

        Starting from the posterior state at index ``K_ctx - 1``, each imagined
        step applies the next action to produce ``(h_{t+1}, prior_mean_{t+1})``
        and feeds that prior mean back as the next stochastic state — i.e. a
        deterministic mean-based prior rollout whose decoded frames are the
        target of the foreground imagination loss.

        The imagined reconstruction at imagined-index ``k`` targets
        ``next_observations[:, K_ctx - 1 + k]`` which is ``obs_{K_ctx + k}``.

        Args:
            rssm_output: Output of a posterior pass, providing ``h`` and the
                posterior mean at every step.
            actions: Full action tensor shaped ``[B, T, A]``. The rollout
                consumes ``actions[:, K_ctx - 1 : K_ctx - 1 + horizon]``.
            context_steps: Number of posterior warmup steps ``K_ctx``. Must be
                at least 1.
            horizon: Number of imagined prior steps ``K_imag``. Must be at
                least 1.

        Returns:
            ``(imagined_reconstructions, imagined_features, target_start)``.
            The reconstruction tensor is shaped
            ``[B, horizon, input_channels, image_size, image_size]``; features
            are ``[B, horizon, hidden_size + latent_size]``; ``target_start``
            is ``K_ctx - 1`` (the index into ``next_observations`` of the
            first imagined target).
        """

        if context_steps < 1:
            raise ValueError("imagination_context_steps must be >= 1 when horizon > 0")
        if horizon < 1:
            raise ValueError("imagination_horizon must be >= 1")
        sequence_length = actions.shape[1]
        if sequence_length < context_steps + horizon - 1:
            raise ValueError(
                "imagination requires sequence_length >= context_steps + horizon - 1; "
                f"got T={sequence_length}, K_ctx={context_steps}, K_imag={horizon}",
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"expected action_dim={self.action_dim}, got {actions.shape[-1]}")

        h_start = rssm_output.deter_states[:, context_steps - 1]
        z_start = rssm_output.posterior_mean[:, context_steps - 1]
        rollout_actions = actions[:, context_steps - 1 : context_steps - 1 + horizon]
        imagined_features = self.imagine_rollout_features(h_start, z_start, rollout_actions)
        imagined_reconstructions = self.decoder(imagined_features)
        target_start = context_steps - 1
        return imagined_reconstructions, imagined_features, target_start

    def imagine_rollout_features(
        self,
        deter_start: torch.Tensor,
        stoch_start: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Roll the prior forward for ``actions.shape[1]`` steps from ``(h, z)``.

        Returns features ``[B, K, H + Z]``. Re-used by ``imagine_from_posterior``
        and by the CEM planner, which calls this with ``B = n_samples`` and
        differing action sequences sharing the same start state. No decoder /
        reward-head invocation here so callers can choose what to compute.
        """

        if deter_start.ndim != 2 or deter_start.shape[-1] != self.hidden_size:
            raise ValueError(
                f"deter_start must have shape [B, hidden_size={self.hidden_size}]; "
                f"got {tuple(deter_start.shape)}",
            )
        if stoch_start.ndim != 2 or stoch_start.shape[-1] != self.latent_size:
            raise ValueError(
                f"stoch_start must have shape [B, latent_size={self.latent_size}]; "
                f"got {tuple(stoch_start.shape)}",
            )
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must have shape [B, K, action_dim={self.action_dim}]; "
                f"got {tuple(actions.shape)}",
            )
        if actions.shape[0] != deter_start.shape[0]:
            raise ValueError("actions batch size must match deter_start batch size")

        h_t = deter_start
        z_t = stoch_start
        horizon = actions.shape[1]
        features_list: list[torch.Tensor] = []
        for step in range(horizon):
            action_t = actions[:, step]
            recurrent_input = torch.cat([z_t, action_t], dim=-1)
            h_t = self.rssm.recurrent(recurrent_input, h_t)
            prior_mean, _ = self.rssm._dist_params(self.rssm.prior(h_t))
            features_list.append(torch.cat([h_t, prior_mean], dim=-1))
            z_t = prior_mean
        return torch.stack(features_list, dim=1)

    def posterior_step(
        self,
        deter_prev: torch.Tensor,
        stoch_prev: torch.Tensor,
        action_prev: torch.Tensor,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one posterior step: ``(h_{t-1}, z_{t-1}, action_{t-1}, obs_t) → (h_t, z_t_mean)``.

        Used by the online MPC loop (and other rollout drivers) to drop the
        next observation into the latent state without re-encoding the entire
        history. Returns deterministic posterior mean (no sampling) for
        reproducibility.
        """

        if observation.ndim == 3:
            observation = observation.unsqueeze(0)
        if observation.ndim != 4 or observation.shape[1] != self.input_channels:
            raise ValueError(
                f"observation must have shape [B, C={self.input_channels}, H, W]; "
                f"got {tuple(observation.shape)}",
            )
        if action_prev.ndim != 2 or action_prev.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_prev must have shape [B, action_dim={self.action_dim}]; "
                f"got {tuple(action_prev.shape)}",
            )

        recurrent_input = torch.cat([stoch_prev, action_prev], dim=-1)
        h_t = self.rssm.recurrent(recurrent_input, deter_prev)
        embedding = self.encoder(observation.unsqueeze(1)).squeeze(1)
        posterior_input = torch.cat([h_t, embedding], dim=-1)
        posterior_mean, _ = self.rssm._dist_params(self.rssm.posterior(posterior_input))
        return h_t, posterior_mean

    def initial_posterior(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initial posterior at the first observation, with zero deterministic state.

        Mirrors the RSSM's ``t=0`` step: ``h_0 = 0``, ``q(z_0 | h_0, embed_0)``.
        Returns ``(h_0, z_0_mean)`` shaped ``[B, hidden_size]``, ``[B, latent_size]``.
        """

        if observation.ndim == 3:
            observation = observation.unsqueeze(0)
        if observation.ndim != 4 or observation.shape[1] != self.input_channels:
            raise ValueError(
                f"observation must have shape [B, C={self.input_channels}, H, W]; "
                f"got {tuple(observation.shape)}",
            )
        batch_size = observation.shape[0]
        h_0 = observation.new_zeros(batch_size, self.hidden_size)
        embedding = self.encoder(observation.unsqueeze(1)).squeeze(1)
        posterior_input = torch.cat([h_0, embedding], dim=-1)
        posterior_mean, _ = self.rssm._dist_params(self.rssm.posterior(posterior_input))
        return h_0, posterior_mean

    def transition_aligned_next_prior_features(
        self,
        embeddings: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next prior features for ``obs_t --action_t--> obs_{t+1}``.

        Args:
            embeddings: Current observation embeddings shaped ``[B, T, E]``.
            actions: Action tensor shaped ``[B, T, A]``.

        Returns:
            Predicted next prior features shaped ``[B, T, H + Z]``.
        """

        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [B, T, E]")
        if actions.ndim != 3:
            raise ValueError("actions must have shape [B, T, A]")
        if embeddings.shape[:2] != actions.shape[:2]:
            raise ValueError("embeddings and actions must share [B, T]")
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"expected action_dim={self.action_dim}, got {actions.shape[-1]}")

        batch_size, sequence_length = embeddings.shape[:2]
        device = embeddings.device
        dtype = embeddings.dtype
        h_t = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        predicted_next_features: list[torch.Tensor] = []

        for step in range(sequence_length):
            # q(z_t | h_t, embed_t): posterior_mean is [B, Z].
            posterior_input = torch.cat([h_t, embeddings[:, step]], dim=-1)
            posterior_mean, _ = self.rssm._dist_params(self.rssm.posterior(posterior_input))

            # action_t advances the deterministic state to h_{t+1}; the prior
            # mean predicts z_{t+1} before seeing obs_{t+1}.
            recurrent_input = torch.cat([posterior_mean, actions[:, step]], dim=-1)
            h_t = self.rssm.recurrent(recurrent_input, h_t)
            prior_next_mean, _ = self.rssm._dist_params(self.rssm.prior(h_t))
            predicted_next_features.append(torch.cat([h_t, prior_next_mean], dim=-1))

        return torch.stack(predicted_next_features, dim=1)
