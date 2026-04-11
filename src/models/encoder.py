"""Compact convolutional encoder for pixel observations."""

from __future__ import annotations

import torch
from torch import nn


class ConvEncoder(nn.Module):
    """Encode image sequences from ``[B, T, 3, 84, 84]`` to ``[B, T, E]``."""

    def __init__(self, embedding_size: int, input_channels: int = 3) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.input_channels = input_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Linear(128 * 5 * 5, embedding_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations.

        Args:
            observations: Float tensor shaped ``[B, T, C, H, W]``.

        Returns:
            Embeddings shaped ``[B, T, E]``.
        """

        if observations.ndim != 5:
            raise ValueError("observations must have shape [B, T, C, H, W]")

        batch_size, sequence_length = observations.shape[:2]
        frames = observations.reshape(batch_size * sequence_length, *observations.shape[2:])
        encoded = self.cnn(frames)
        embeddings = self.projection(encoded.flatten(start_dim=1))
        return embeddings.reshape(batch_size, sequence_length, self.embedding_size)
