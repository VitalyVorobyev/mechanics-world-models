"""Minimal recurrent state-space model with Gaussian latent states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class RSSMOutput:
    """RSSM rollout tensors, all time-major dimensions kept as ``[B, T, ...]``."""

    prior_mean: torch.Tensor
    prior_std: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_std: torch.Tensor
    latents: torch.Tensor
    deter_states: torch.Tensor

    @property
    def features(self) -> torch.Tensor:
        """Concatenate deterministic and sampled stochastic states as ``[B, T, H + Z]``.

        Uses the reparameterized sample ``z_t``. Prefer ``features_posterior_mean``
        for decoder and reward-head inputs during training so that the decoded
        distribution matches the deterministic mean-based rollout used at eval
        time (eliminates the train/eval mean-vs-sample mismatch that otherwise
        leaves the decoder under-trained for the exact features fed into it at
        evaluation).
        """

        return torch.cat([self.deter_states, self.latents], dim=-1)

    @property
    def features_posterior_mean(self) -> torch.Tensor:
        """Concatenate deterministic and posterior-mean states as ``[B, T, H + Z]``.

        Used as the decoder input during training so that the decoder is trained
        on the same posterior representation (``h_t``, ``posterior_mean_t``) that
        evaluation uses for reconstruction.
        """

        return torch.cat([self.deter_states, self.posterior_mean], dim=-1)


class RSSM(nn.Module):
    """Action-conditioned recurrent state-space model."""

    def __init__(
        self,
        action_dim: int,
        embedding_size: int,
        hidden_size: int,
        latent_size: int,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.min_std = min_std

        self.recurrent = nn.GRUCell(latent_size + action_dim, hidden_size)
        self.prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * latent_size),
        )
        self.posterior = nn.Sequential(
            nn.Linear(hidden_size + embedding_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * latent_size),
        )

    def forward(self, embeddings: torch.Tensor, actions: torch.Tensor) -> RSSMOutput:
        """Run posterior inference over a sequence.

        Args:
            embeddings: Observation embeddings shaped ``[B, T, E]``.
            actions: Action sequence shaped ``[B, T, A]``.

        Returns:
            RSSMOutput with prior/posterior parameters, sampled ``z_t``, and
            deterministic states ``h_t`` all shaped ``[B, T, ...]``.
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
        z_t = torch.zeros(batch_size, self.latent_size, device=device, dtype=dtype)

        prior_means: list[torch.Tensor] = []
        prior_stds: list[torch.Tensor] = []
        posterior_means: list[torch.Tensor] = []
        posterior_stds: list[torch.Tensor] = []
        latents: list[torch.Tensor] = []
        deter_states: list[torch.Tensor] = []

        for step in range(sequence_length):
            # h_t is the deterministic state for obs_t. At t=0 it is zero; at
            # later steps it has already been advanced by action_{t-1}.
            # Prior p(z_t | h_t): each parameter tensor is [B, Z].
            prior_mean, prior_std = self._dist_params(self.prior(h_t))

            # Posterior q(z_t | h_t, embed_t): concat is [B, H + E].
            posterior_input = torch.cat([h_t, embeddings[:, step]], dim=-1)
            posterior_mean, posterior_std = self._dist_params(self.posterior(posterior_input))
            z_t = self._sample(posterior_mean, posterior_std)

            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            posterior_means.append(posterior_mean)
            posterior_stds.append(posterior_std)
            latents.append(z_t)
            deter_states.append(h_t)

            # action_t advances obs_t to obs_{t+1}; this h_{t+1} is used by the
            # next loop iteration and by transition-aligned prediction helpers.
            recurrent_input = torch.cat([z_t, actions[:, step]], dim=-1)
            h_t = self.recurrent(recurrent_input, h_t)

        return RSSMOutput(
            prior_mean=torch.stack(prior_means, dim=1),
            prior_std=torch.stack(prior_stds, dim=1),
            posterior_mean=torch.stack(posterior_means, dim=1),
            posterior_std=torch.stack(posterior_stds, dim=1),
            latents=torch.stack(latents, dim=1),
            deter_states=torch.stack(deter_states, dim=1),
        )

    def _dist_params(self, raw_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = torch.chunk(raw_params, chunks=2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return mean, std

    @staticmethod
    def _sample(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(mean)
        return mean + std * noise
