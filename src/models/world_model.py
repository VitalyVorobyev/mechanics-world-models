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


class VisualWorldModel(nn.Module):
    """Encode observations, infer RSSM states, and decode reconstructions."""

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

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> WorldModelOutput:
        """Run a posterior world-model pass.

        Args:
            observations: Images shaped ``[B, T, 3, 84, 84]``.
            actions: Actions shaped ``[B, T, A]``.

        Returns:
            Embeddings ``[B, T, E]``, RSSM tensors, and reconstructions
            ``[B, T, 3, 84, 84]``.
        """

        embeddings = self.encoder(observations)
        rssm_output = self.rssm(embeddings=embeddings, actions=actions)
        reconstructions = self.decoder(rssm_output.features)
        return WorldModelOutput(
            embeddings=embeddings,
            rssm=rssm_output,
            reconstructions=reconstructions,
        )
