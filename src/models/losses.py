"""Losses for the minimal visual world model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.world_model import WorldModelOutput


def compute_world_model_loss(
    outputs: WorldModelOutput,
    observations: torch.Tensor,
    kl_weight: float,
    reconstruction_weight: float = 1.0,
    foreground_reconstruction_weight: float = 0.0,
    foreground_mask_floor: float = 0.02,
    foreground_mask_kernel_size: int = 7,
    transition_reconstruction_weight: float = 0.0,
    foreground_transition_reconstruction_weight: float = 0.0,
    latent_consistency_weight: float = 0.0,
    next_observations: torch.Tensor | None = None,
    dynamic_reconstruction_weight: float = 0.0,
    dynamic_mask_floor: float = 0.05,
    kl_balance_alpha: float = 0.8,
    kl_free_nats: float = 3.0,
    imagination_reconstruction_weight: float = 0.0,
    foreground_imagination_reconstruction_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute reconstruction, RSSM KL, and optional transition/latent losses.

    Args:
        outputs: World-model outputs for a batch.
        observations: Target images shaped ``[B, T, 3, 84, 84]``.
        kl_weight: Scalar multiplier for the KL term.
        reconstruction_weight: Scalar multiplier for the pixel reconstruction term.
        foreground_reconstruction_weight: Multiplier for background-subtraction MSE.
        foreground_mask_floor: Minimum unnormalized mask value for static pixels.
        foreground_mask_kernel_size: Odd max-pool kernel size for foreground masks.
        transition_reconstruction_weight: Multiplier for full-frame next-frame MSE.
        foreground_transition_reconstruction_weight: Multiplier for masked next-frame MSE.
        latent_consistency_weight: Scalar multiplier for next-latent consistency.
        next_observations: Optional next images shaped ``[B, T, 3, 84, 84]``.
        dynamic_reconstruction_weight: Scalar multiplier for motion-weighted MSE.
        dynamic_mask_floor: Minimum unnormalized mask value for static pixels.
        kl_balance_alpha: DreamerV2-style balancing coefficient. ``alpha`` weights
            ``KL(sg(q) || p)`` (train prior), ``1 - alpha`` weights
            ``KL(q || sg(p))`` (regularize posterior). ``alpha=0.5`` recovers
            symmetric KL; ``alpha=1.0`` trains only the prior.
        kl_free_nats: Per-sample-per-step floor in nats. Elements of the
            balanced KL below this floor are clamped so the KL stops pushing
            toward zero once the information budget is met.

    Returns:
        Scalar tensor metrics: ``total_loss``, ``reconstruction_loss``,
        ``weighted_reconstruction_loss``, ``dynamic_reconstruction_loss``,
        ``weighted_dynamic_reconstruction_loss``, ``foreground_reconstruction_loss``,
        ``weighted_foreground_reconstruction_loss``, transition reconstruction losses,
        ``kl_loss`` (balanced + free-nats clamped, the value gradients flow through),
        ``weighted_kl_loss``, ``kl_raw`` (unbalanced ``KL(q || p)`` mean, diagnostic),
        ``kl_balanced_raw`` (balanced KL before clamp, diagnostic),
        ``kl_free_nats_active`` (fraction of samples at or below the floor),
        ``latent_consistency_loss``, and ``weighted_latent_consistency_loss``.
    """

    # Reconstruction targets and decoder outputs are aligned at the same
    # timestep: both tensors are [B, T, 3, 84, 84] floats in [0, 1].
    reconstruction_loss = F.mse_loss(outputs.reconstructions, observations)
    weighted_reconstruction_loss = reconstruction_weight * reconstruction_loss
    dynamic_reconstruction_loss = compute_dynamic_reconstruction_loss(
        outputs=outputs,
        observations=observations,
        next_observations=next_observations,
        dynamic_reconstruction_weight=dynamic_reconstruction_weight,
        dynamic_mask_floor=dynamic_mask_floor,
    )
    weighted_dynamic_reconstruction_loss = (
        dynamic_reconstruction_weight * dynamic_reconstruction_loss
    )
    foreground_reconstruction_loss = compute_foreground_reconstruction_loss(
        outputs=outputs,
        observations=observations,
        foreground_reconstruction_weight=foreground_reconstruction_weight,
        foreground_mask_floor=foreground_mask_floor,
        foreground_mask_kernel_size=foreground_mask_kernel_size,
    )
    weighted_foreground_reconstruction_loss = (
        foreground_reconstruction_weight * foreground_reconstruction_loss
    )
    transition_reconstruction_loss, foreground_transition_reconstruction_loss = (
        compute_transition_reconstruction_losses(
            outputs=outputs,
            next_observations=next_observations,
            transition_reconstruction_weight=transition_reconstruction_weight,
            foreground_transition_reconstruction_weight=foreground_transition_reconstruction_weight,
            foreground_mask_floor=foreground_mask_floor,
            foreground_mask_kernel_size=foreground_mask_kernel_size,
        )
    )
    weighted_transition_reconstruction_loss = (
        transition_reconstruction_weight * transition_reconstruction_loss
    )
    weighted_foreground_transition_reconstruction_loss = (
        foreground_transition_reconstruction_weight * foreground_transition_reconstruction_loss
    )
    imagination_reconstruction_loss, foreground_imagination_reconstruction_loss = (
        compute_imagination_reconstruction_losses(
            outputs=outputs,
            next_observations=next_observations,
            imagination_reconstruction_weight=imagination_reconstruction_weight,
            foreground_imagination_reconstruction_weight=(
                foreground_imagination_reconstruction_weight
            ),
            foreground_mask_floor=foreground_mask_floor,
            foreground_mask_kernel_size=foreground_mask_kernel_size,
        )
    )
    weighted_imagination_reconstruction_loss = (
        imagination_reconstruction_weight * imagination_reconstruction_loss
    )
    weighted_foreground_imagination_reconstruction_loss = (
        foreground_imagination_reconstruction_weight * foreground_imagination_reconstruction_loss
    )
    kl_parts = compute_rssm_kl_loss(
        posterior_mean=outputs.rssm.posterior_mean,
        posterior_std=outputs.rssm.posterior_std,
        prior_mean=outputs.rssm.prior_mean,
        prior_std=outputs.rssm.prior_std,
        kl_balance_alpha=kl_balance_alpha,
        kl_free_nats=kl_free_nats,
    )
    kl_loss = kl_parts["kl_loss"]
    weighted_kl_loss = kl_weight * kl_loss
    latent_consistency_loss = compute_latent_consistency_loss(outputs, latent_consistency_weight)
    weighted_latent_consistency_loss = latent_consistency_weight * latent_consistency_loss
    total_loss = (
        weighted_reconstruction_loss
        + weighted_dynamic_reconstruction_loss
        + weighted_foreground_reconstruction_loss
        + weighted_transition_reconstruction_loss
        + weighted_foreground_transition_reconstruction_loss
        + weighted_imagination_reconstruction_loss
        + weighted_foreground_imagination_reconstruction_loss
        + weighted_kl_loss
        + weighted_latent_consistency_loss
    )
    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "weighted_reconstruction_loss": weighted_reconstruction_loss,
        "dynamic_reconstruction_loss": dynamic_reconstruction_loss,
        "weighted_dynamic_reconstruction_loss": weighted_dynamic_reconstruction_loss,
        "foreground_reconstruction_loss": foreground_reconstruction_loss,
        "weighted_foreground_reconstruction_loss": weighted_foreground_reconstruction_loss,
        "transition_reconstruction_loss": transition_reconstruction_loss,
        "weighted_transition_reconstruction_loss": weighted_transition_reconstruction_loss,
        "foreground_transition_reconstruction_loss": foreground_transition_reconstruction_loss,
        "weighted_foreground_transition_reconstruction_loss": (
            weighted_foreground_transition_reconstruction_loss
        ),
        "imagination_reconstruction_loss": imagination_reconstruction_loss,
        "weighted_imagination_reconstruction_loss": weighted_imagination_reconstruction_loss,
        "foreground_imagination_reconstruction_loss": foreground_imagination_reconstruction_loss,
        "weighted_foreground_imagination_reconstruction_loss": (
            weighted_foreground_imagination_reconstruction_loss
        ),
        "kl_loss": kl_loss,
        "weighted_kl_loss": weighted_kl_loss,
        "kl_raw": kl_parts["kl_raw"],
        "kl_balanced_raw": kl_parts["kl_balanced_raw"],
        "kl_free_nats_active": kl_parts["kl_free_nats_active"],
        "latent_consistency_loss": latent_consistency_loss,
        "weighted_latent_consistency_loss": weighted_latent_consistency_loss,
    }


def compute_rssm_kl_loss(
    posterior_mean: torch.Tensor,
    posterior_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    kl_balance_alpha: float = 0.8,
    kl_free_nats: float = 3.0,
) -> dict[str, torch.Tensor]:
    """DreamerV2-style KL-balanced loss with a free-nats floor.

    Computes ``alpha * KL(sg(q) || p) + (1 - alpha) * KL(q || sg(p))`` per
    sample and step (summed over latent_size), clamps the floor, and averages.
    The returned ``kl_loss`` is the quantity gradients should flow through;
    the additional tensors are diagnostic only.
    """

    if not 0.0 <= kl_balance_alpha <= 1.0:
        raise ValueError("kl_balance_alpha must satisfy 0 <= alpha <= 1")
    if kl_free_nats < 0.0:
        raise ValueError("kl_free_nats must be >= 0")

    # KL(sg(q) || p): gradient flows only into the prior. Trains the prior to
    # match the current posterior without pulling the posterior toward a still-
    # random prior, which is the DreamerV2 rationale for asymmetric balancing.
    kl_post_term = diagonal_gaussian_kl(
        mean_q=posterior_mean.detach(),
        std_q=posterior_std.detach(),
        mean_p=prior_mean,
        std_p=prior_std,
    )
    # KL(q || sg(p)): gradient flows only into the posterior. Regularizes the
    # posterior toward the (detached) prior so the posterior cannot encode
    # information the prior has no hope of predicting.
    kl_prior_term = diagonal_gaussian_kl(
        mean_q=posterior_mean,
        std_q=posterior_std,
        mean_p=prior_mean.detach(),
        std_p=prior_std.detach(),
    )
    # Both tensors are [B, T]; KL already summed over latent_size.
    kl_balanced = kl_balance_alpha * kl_post_term + (1.0 - kl_balance_alpha) * kl_prior_term
    kl_clamped = torch.clamp(kl_balanced, min=kl_free_nats)
    kl_loss = kl_clamped.mean()

    with torch.no_grad():
        kl_raw = diagonal_gaussian_kl(
            mean_q=posterior_mean,
            std_q=posterior_std,
            mean_p=prior_mean,
            std_p=prior_std,
        ).mean()
        kl_balanced_raw = kl_balanced.detach().mean()
        kl_free_nats_active = (kl_balanced.detach() <= kl_free_nats).float().mean()

    return {
        "kl_loss": kl_loss,
        "kl_raw": kl_raw,
        "kl_balanced_raw": kl_balanced_raw,
        "kl_free_nats_active": kl_free_nats_active,
    }


def compute_imagination_reconstruction_losses(
    outputs: WorldModelOutput,
    next_observations: torch.Tensor | None,
    imagination_reconstruction_weight: float,
    foreground_imagination_reconstruction_weight: float,
    foreground_mask_floor: float,
    foreground_mask_kernel_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute decoded-imagination losses against the aligned next-observation slice.

    ``outputs.imagined_reconstructions[:, k]`` is decoded from the prior
    feature reached after applying ``action_{K_ctx - 1 + k}`` to the posterior
    state at index ``K_ctx - 1``. It targets
    ``next_observations[:, K_ctx - 1 + k]`` which equals ``obs_{K_ctx + k}``.
    Both tensors are ``[B, K_imag, 3, 84, 84]`` floats in ``[0, 1]``.
    """

    if imagination_reconstruction_weight < 0.0:
        raise ValueError("imagination_reconstruction_weight must be >= 0")
    if foreground_imagination_reconstruction_weight < 0.0:
        raise ValueError("foreground_imagination_reconstruction_weight must be >= 0")

    zero = outputs.reconstructions.new_zeros(())
    if (
        imagination_reconstruction_weight == 0.0
        and foreground_imagination_reconstruction_weight == 0.0
    ):
        return zero, zero
    if outputs.imagined_reconstructions is None or outputs.imagined_target_start is None:
        raise ValueError(
            "imagination_context_steps and imagination_horizon must be passed to the "
            "model when imagination reconstruction weights are > 0",
        )
    if next_observations is None:
        raise ValueError(
            "next_observations must be passed to the loss when imagination "
            "reconstruction weights are > 0",
        )

    imagined = outputs.imagined_reconstructions
    target_start = outputs.imagined_target_start
    horizon = imagined.shape[1]
    if next_observations.shape[1] < target_start + horizon:
        raise ValueError(
            "next_observations too short for imagination targets: "
            f"need index up to {target_start + horizon - 1}, "
            f"got T={next_observations.shape[1]}",
        )
    target = next_observations[:, target_start : target_start + horizon]
    if imagined.shape != target.shape:
        raise ValueError(
            f"imagination target shape mismatch: {imagined.shape} vs {target.shape}",
        )

    squared_error = (imagined - target).pow(2)
    imagination_loss = (
        squared_error.mean() if imagination_reconstruction_weight > 0.0 else zero
    )
    if foreground_imagination_reconstruction_weight > 0.0:
        mask = foreground_reconstruction_mask(
            observations=target,
            floor=foreground_mask_floor,
            kernel_size=foreground_mask_kernel_size,
        )
        foreground_imagination_loss = (squared_error * mask.detach()).mean()
    else:
        foreground_imagination_loss = zero
    return imagination_loss, foreground_imagination_loss


def compute_transition_reconstruction_losses(
    outputs: WorldModelOutput,
    next_observations: torch.Tensor | None,
    transition_reconstruction_weight: float,
    foreground_transition_reconstruction_weight: float,
    foreground_mask_floor: float,
    foreground_mask_kernel_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute opt-in next-frame reconstruction losses.

    ``outputs.next_reconstructions[:, t]`` is decoded from the prior feature
    after applying ``action_t`` and therefore targets ``next_observations[:, t]``
    / ``obs_{t+1}``. Both image tensors are ``[B, T, 3, 84, 84]`` floats in
    ``[0, 1]``.
    """

    if transition_reconstruction_weight < 0.0:
        raise ValueError("transition_reconstruction_weight must be >= 0")
    if foreground_transition_reconstruction_weight < 0.0:
        raise ValueError("foreground_transition_reconstruction_weight must be >= 0")

    zero = outputs.reconstructions.new_zeros(())
    needs_transition_target = (
        transition_reconstruction_weight > 0.0
        or foreground_transition_reconstruction_weight > 0.0
    )
    if not needs_transition_target:
        return zero, zero
    if next_observations is None:
        raise ValueError(
            "next_observations must be passed to the loss when transition "
            "reconstruction weights are > 0",
        )
    if outputs.next_reconstructions is None:
        raise ValueError(
            "next_observations must be passed to the model when transition "
            "reconstruction weights are > 0",
        )
    if outputs.next_reconstructions.shape != next_observations.shape:
        raise ValueError(
            "transition reconstruction shape mismatch: "
            f"{outputs.next_reconstructions.shape} != {next_observations.shape}",
        )

    squared_error = (outputs.next_reconstructions - next_observations).pow(2)
    transition_loss = (
        squared_error.mean() if transition_reconstruction_weight > 0.0 else zero
    )

    if foreground_transition_reconstruction_weight > 0.0:
        mask = foreground_reconstruction_mask(
            observations=next_observations,
            floor=foreground_mask_floor,
            kernel_size=foreground_mask_kernel_size,
        )
        foreground_transition_loss = (squared_error * mask.detach()).mean()
    else:
        foreground_transition_loss = zero
    return transition_loss, foreground_transition_loss


def compute_foreground_reconstruction_loss(
    outputs: WorldModelOutput,
    observations: torch.Tensor,
    foreground_reconstruction_weight: float,
    foreground_mask_floor: float,
    foreground_mask_kernel_size: int,
) -> torch.Tensor:
    """Compute background-subtraction weighted reconstruction MSE."""

    if foreground_reconstruction_weight < 0.0:
        raise ValueError("foreground_reconstruction_weight must be >= 0")
    if foreground_reconstruction_weight == 0.0:
        return outputs.reconstructions.new_zeros(())
    if outputs.reconstructions.shape != observations.shape:
        raise ValueError(
            f"reconstruction shape mismatch: {outputs.reconstructions.shape} != {observations.shape}",
        )

    mask = foreground_reconstruction_mask(
        observations=observations,
        floor=foreground_mask_floor,
        kernel_size=foreground_mask_kernel_size,
    )
    squared_error = (outputs.reconstructions - observations).pow(2)
    return (squared_error * mask.detach()).mean()


def foreground_reconstruction_mask(
    observations: torch.Tensor,
    floor: float = 0.02,
    kernel_size: int = 7,
) -> torch.Tensor:
    """Build a normalized background-subtraction mask ``[B, T, 1, H, W]``.

    The batch background is ``mean(observations, dim=(0, 1))`` with gradients
    stopped. Saliency is ``abs(obs_t - background)`` averaged over RGB channels,
    max-pooled around object edges, floored, and normalized to mean one per
    frame. This keeps the loss scale comparable to full-frame MSE.
    """

    if not 0.0 <= floor <= 1.0:
        raise ValueError("foreground_mask_floor must satisfy 0 <= floor <= 1")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("foreground_mask_kernel_size must be a positive odd integer")

    saliency = foreground_saliency(observations)
    batch_size, sequence_length, _, height, width = saliency.shape
    flat_saliency = saliency.reshape(batch_size * sequence_length, 1, height, width)
    pooled_saliency = F.max_pool2d(
        flat_saliency,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).reshape_as(saliency)

    saliency_scale = pooled_saliency.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    normalized_saliency = pooled_saliency / saliency_scale
    mask = floor + (1.0 - floor) * normalized_saliency
    mean_mask = mask.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return mask / mean_mask


def foreground_saliency(observations: torch.Tensor) -> torch.Tensor:
    """Return background-subtraction saliency shaped ``[B, T, 1, H, W]``."""

    if observations.ndim != 5:
        raise ValueError(f"observations must have shape [B, T, C, H, W], got {observations.shape}")
    detached_observations = observations.detach()
    background = detached_observations.mean(dim=(0, 1), keepdim=True)
    return (detached_observations - background).abs().mean(dim=2, keepdim=True)


def compute_dynamic_reconstruction_loss(
    outputs: WorldModelOutput,
    observations: torch.Tensor,
    next_observations: torch.Tensor | None,
    dynamic_reconstruction_weight: float,
    dynamic_mask_floor: float,
) -> torch.Tensor:
    """Compute motion-weighted reconstruction MSE from frame differences."""

    if dynamic_reconstruction_weight < 0.0:
        raise ValueError("dynamic_reconstruction_weight must be >= 0")
    if dynamic_reconstruction_weight == 0.0:
        return outputs.reconstructions.new_zeros(())
    if next_observations is None:
        raise ValueError(
            "next_observations must be passed to the loss when "
            "dynamic_reconstruction_weight > 0",
        )
    if observations.shape != next_observations.shape:
        raise ValueError(
            f"observation shape mismatch: {observations.shape} != {next_observations.shape}",
        )
    if outputs.reconstructions.shape != observations.shape:
        raise ValueError(
            f"reconstruction shape mismatch: {outputs.reconstructions.shape} != {observations.shape}",
        )

    mask = dynamic_reconstruction_mask(
        observations=observations,
        next_observations=next_observations,
        floor=dynamic_mask_floor,
    )
    squared_error = (outputs.reconstructions - observations).pow(2)
    return (squared_error * mask.detach()).mean()


def dynamic_reconstruction_mask(
    observations: torch.Tensor,
    next_observations: torch.Tensor,
    floor: float = 0.05,
    kernel_size: int = 7,
) -> torch.Tensor:
    """Build a normalized motion mask shaped ``[B, T, 1, H, W]``.

    The mask is derived from ``abs(obs_{t+1} - obs_t)`` and max-pooled to cover
    nearby pixels around motion edges. It is normalized to mean one per frame so
    the dynamic reconstruction loss stays on roughly the same scale as full-frame
    MSE while emphasizing moving regions.
    """

    if observations.shape != next_observations.shape:
        raise ValueError(
            f"observation shape mismatch: {observations.shape} != {next_observations.shape}",
        )
    if observations.ndim != 5:
        raise ValueError(f"observations must have shape [B, T, C, H, W], got {observations.shape}")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("dynamic_mask_floor must satisfy 0 <= floor <= 1")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    # motion: [B, T, 1, H, W], in observation pixel units.
    motion = (next_observations - observations).abs().mean(dim=2, keepdim=True)
    batch_size, sequence_length, _, height, width = motion.shape
    flat_motion = motion.reshape(batch_size * sequence_length, 1, height, width)
    pooled_motion = F.max_pool2d(
        flat_motion,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).reshape_as(motion)

    motion_scale = pooled_motion.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    normalized_motion = pooled_motion / motion_scale
    mask = floor + (1.0 - floor) * normalized_motion
    mean_mask = mask.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return mask / mean_mask


def compute_latent_consistency_loss(
    outputs: WorldModelOutput,
    latent_consistency_weight: float,
) -> torch.Tensor:
    """Compute BYOL-style cosine loss against stop-gradient next embeddings."""

    predicted = outputs.predicted_next_embeddings
    target = outputs.next_embeddings
    if predicted is None or target is None:
        if latent_consistency_weight > 0.0:
            raise ValueError(
                "next_observations must be passed to the model when "
                "latent_consistency_weight > 0",
            )
        return outputs.reconstructions.new_zeros(())
    if predicted.shape != target.shape:
        raise ValueError(f"latent shape mismatch: {predicted.shape} != {target.shape}")

    # predicted and target: [B, T, E]. The target is detached here even though
    # VisualWorldModel also detaches it, so the loss contract is explicit.
    cosine = F.cosine_similarity(predicted, target.detach(), dim=-1)
    return (1.0 - cosine).mean()


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
