"""Losses for the minimal visual world model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.world_model import WorldModelOutput


def compute_world_model_loss(
    outputs: WorldModelOutput,
    observations: torch.Tensor,
    kl_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute reconstruction and RSSM KL losses.

    Args:
        outputs: World-model outputs for a batch.
        observations: Target images shaped ``[B, T, 3, 84, 84]``.
        kl_weight: Scalar multiplier for the KL term.

    Returns:
        Scalar tensor metrics: ``total_loss``, ``reconstruction_loss``,
        ``kl_loss``, and ``weighted_kl_loss``.
    """

    reconstruction_loss = F.mse_loss(outputs.reconstructions, observations)
    kl_loss = diagonal_gaussian_kl(
        mean_q=outputs.rssm.posterior_mean,
        std_q=outputs.rssm.posterior_std,
        mean_p=outputs.rssm.prior_mean,
        std_p=outputs.rssm.prior_std,
    ).mean()
    weighted_kl_loss = kl_weight * kl_loss
    total_loss = reconstruction_loss + weighted_kl_loss
    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
        "weighted_kl_loss": weighted_kl_loss,
    }


def diagonal_gaussian_kl(
    mean_q: torch.Tensor,
    std_q: torch.Tensor,
    mean_p: torch.Tensor,
    std_p: torch.Tensor,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians, reduced over latent size only.

    Inputs are shaped ``[B, T, Z]`` and the returned tensor is ``[B, T]``.
    """

    var_q = std_q.pow(2)
    var_p = std_p.pow(2)
    kl_per_dim = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p).pow(2)) / (2.0 * var_p) - 0.5
    return kl_per_dim.sum(dim=-1)
