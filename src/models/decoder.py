"""Compact image decoder for RSSM latent features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ImageDecoder(nn.Module):
    """Decode latent features from ``[B, T, F]`` to ``[B, T, 3, 84, 84]``."""

    def __init__(
        self,
        feature_size: int,
        output_channels: int = 3,
        output_size: int = 84,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.output_channels = output_channels
        self.output_size = output_size
        self.projection = nn.Sequential(
            nn.Linear(feature_size, 128 * 5 * 5),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features.

        Args:
            features: Latent feature tensor shaped ``[B, T, F]``.

        Returns:
            Reconstructed observations shaped ``[B, T, C, H, W]`` in ``[0, 1]``.
        """

        if features.ndim != 3:
            raise ValueError("features must have shape [B, T, F]")

        batch_size, sequence_length = features.shape[:2]
        flat_features = features.reshape(batch_size * sequence_length, self.feature_size)
        hidden = self.projection(flat_features).reshape(batch_size * sequence_length, 128, 5, 5)
        decoded = self.deconv(hidden)
        if decoded.shape[-2:] != (self.output_size, self.output_size):
            decoded = F.interpolate(
                decoded,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        reconstructions = torch.sigmoid(decoded)
        return reconstructions.reshape(
            batch_size,
            sequence_length,
            self.output_channels,
            self.output_size,
            self.output_size,
        )
