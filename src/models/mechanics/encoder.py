"""Factored encoder producing ``(q, qdot, z_nuisance)`` posteriors.

The visible factorization bias is weak by design — a shared conv trunk feeds
two heads, one outputting the mechanics branch ``(q, qdot)`` and one the
nuisance branch ``z``. The gradient signal that forces the split to mean
anything comes from (a) the Lagrangian prior being applied *only* to the
mechanics branch and (b) the smoothness loss tying ``qdot`` to the
finite-difference of ``q``. Without those training objectives the encoder
would happily put everything into whichever branch has easier gradients, so
this module on its own guarantees nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from models.encoder import ConvEncoder


@dataclass(frozen=True)
class FactoredPosterior:
    """Per-timestep posterior parameters for the mechanics+nuisance split."""

    q_mean: torch.Tensor  # [B, T, d]
    q_std: torch.Tensor  # [B, T, d]
    qdot_mean: torch.Tensor  # [B, T, d]
    qdot_std: torch.Tensor  # [B, T, d]
    z_mean: torch.Tensor  # [B, T, k]
    z_std: torch.Tensor  # [B, T, k]


class FactoredEncoder(nn.Module):
    """Shared ConvEncoder trunk + two Gaussian heads (mech, nuisance).

    The trunk reuses ``ConvEncoder``'s projection to ``embedding_size``; the
    two heads are small MLPs that emit ``(mean, raw_std)`` for their branch.
    Standard deviations are positivized via ``softplus + min_std`` exactly as
    in the RSSM so eval-time sampling behaves the same way.
    """

    def __init__(
        self,
        q_dim: int,
        nuisance_dim: int,
        embedding_size: int = 256,
        head_hidden_size: int = 128,
        mech_min_std: float = 0.05,
        nuisance_min_std: float = 0.1,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        if q_dim < 1:
            raise ValueError("q_dim must be >= 1")
        if nuisance_dim < 1:
            raise ValueError("nuisance_dim must be >= 1")
        self.q_dim = q_dim
        self.nuisance_dim = nuisance_dim
        self.embedding_size = embedding_size
        self.mech_min_std = mech_min_std
        self.nuisance_min_std = nuisance_min_std

        self.trunk = ConvEncoder(
            embedding_size=embedding_size,
            input_channels=input_channels,
        )
        self.mech_head = nn.Sequential(
            nn.Linear(embedding_size, head_hidden_size),
            nn.ELU(),
            nn.Linear(head_hidden_size, 4 * q_dim),
        )
        self.nuisance_head = nn.Sequential(
            nn.Linear(embedding_size, head_hidden_size),
            nn.ELU(),
            nn.Linear(head_hidden_size, 2 * nuisance_dim),
        )

    def forward(self, observations: torch.Tensor) -> FactoredPosterior:
        """Encode observations to posterior parameters."""

        if observations.ndim != 5:
            raise ValueError("observations must have shape [B, T, C, H, W]")
        embeddings = self.trunk(observations)  # [B, T, E]
        mech_raw = self.mech_head(embeddings)  # [B, T, 4d]
        nuis_raw = self.nuisance_head(embeddings)  # [B, T, 2k]
        q_mean, q_raw_std, qdot_mean, qdot_raw_std = torch.chunk(mech_raw, chunks=4, dim=-1)
        z_mean, z_raw_std = torch.chunk(nuis_raw, chunks=2, dim=-1)
        q_std = F.softplus(q_raw_std) + self.mech_min_std
        qdot_std = F.softplus(qdot_raw_std) + self.mech_min_std
        z_std = F.softplus(z_raw_std) + self.nuisance_min_std
        return FactoredPosterior(
            q_mean=q_mean,
            q_std=q_std,
            qdot_mean=qdot_mean,
            qdot_std=qdot_std,
            z_mean=z_mean,
            z_std=z_std,
        )
