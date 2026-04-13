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
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.image_size = image_size
        self.input_channels = input_channels

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
        reconstructions = self.decoder(rssm_output.features_posterior_mean)
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
        if imagination_horizon > 0 or imagination_context_steps > 0:
            imagined_reconstructions, imagined_features, imagined_target_start = (
                self.imagine_from_posterior(
                    rssm_output=rssm_output,
                    actions=actions,
                    context_steps=imagination_context_steps,
                    horizon=imagination_horizon,
                )
            )

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

        h_t = rssm_output.deter_states[:, context_steps - 1]
        z_t = rssm_output.posterior_mean[:, context_steps - 1]
        features_list: list[torch.Tensor] = []
        for step in range(horizon):
            action_t = actions[:, context_steps - 1 + step]
            recurrent_input = torch.cat([z_t, action_t], dim=-1)
            h_t = self.rssm.recurrent(recurrent_input, h_t)
            prior_mean, _ = self.rssm._dist_params(self.rssm.prior(h_t))
            features_list.append(torch.cat([h_t, prior_mean], dim=-1))
            z_t = prior_mean

        imagined_features = torch.stack(features_list, dim=1)
        imagined_reconstructions = self.decoder(imagined_features)
        target_start = context_steps - 1
        return imagined_reconstructions, imagined_features, target_start

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
