"""Composite training loss for MechanicsWorldModel.

Reuses the tensor-level helpers from ``models.losses`` (foreground mask,
diagonal Gaussian KL) but wires them to the factored ``(q, qdot, z)``
distribution instead of a single flat ``z``. Key differences vs the RSSM
loss:

* Three independent KL terms (``q``, ``qdot``, ``z``) with their own
  free-nats floors — the plan calls for an asymmetric budget where the
  mechanics branch is tighter than the nuisance branch.
* Smoothness prior on ``qdot``: the encoder's ``qdot`` head should equal the
  finite-difference of the ``q`` head so the factorization actually *means*
  generalized coordinate and its time derivative.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.losses import (
    diagonal_gaussian_kl,
    foreground_reconstruction_mask,
)
from models.mechanics.world_model import MechanicsWorldModelOutput


def compute_mechanics_world_model_loss(
    outputs: MechanicsWorldModelOutput,
    observations: torch.Tensor,
    next_observations: torch.Tensor | None = None,
    rewards: torch.Tensor | None = None,
    *,
    reconstruction_weight: float = 1.0,
    foreground_reconstruction_weight: float = 0.0,
    foreground_mask_floor: float = 0.02,
    foreground_mask_kernel_size: int = 7,
    imagination_reconstruction_weight: float = 0.0,
    foreground_imagination_reconstruction_weight: float = 0.0,
    kl_weight: float = 1.0,
    kl_balance_alpha: float = 0.8,
    kl_free_nats_mech: float = 0.5,
    kl_free_nats_nuisance: float = 3.0,
    smoothness_weight: float = 0.0,
    dt: float,
    reward_weight: float = 0.0,
    imagined_reward_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute the mechanics world-model composite loss.

    Gradients flow through ``total_loss``; every other entry is either a
    weighted or a diagnostic tensor. ``dt`` is required because the
    smoothness prior uses it to compare the encoder's ``qdot`` head to the
    finite-difference of the ``q`` head; passing the model's ``dt`` keeps
    the two consistent.
    """

    rollout = outputs.rollout
    posterior = rollout.posterior
    zero = outputs.reconstructions.new_zeros(())

    reconstruction_loss = F.mse_loss(outputs.reconstructions, observations)
    weighted_reconstruction_loss = reconstruction_weight * reconstruction_loss

    foreground_reconstruction_loss = _foreground_mse(
        outputs.reconstructions,
        observations,
        weight=foreground_reconstruction_weight,
        floor=foreground_mask_floor,
        kernel_size=foreground_mask_kernel_size,
        zero=zero,
    )
    weighted_foreground_reconstruction_loss = (
        foreground_reconstruction_weight * foreground_reconstruction_loss
    )

    imagination_loss, foreground_imagination_loss = _imagination_mse(
        imagined=outputs.imagined_reconstructions,
        target_start=outputs.imagined_target_start,
        next_observations=next_observations,
        weight=imagination_reconstruction_weight,
        fg_weight=foreground_imagination_reconstruction_weight,
        foreground_mask_floor=foreground_mask_floor,
        foreground_mask_kernel_size=foreground_mask_kernel_size,
        zero=zero,
    )
    weighted_imagination_loss = imagination_reconstruction_weight * imagination_loss
    weighted_foreground_imagination_loss = (
        foreground_imagination_reconstruction_weight * foreground_imagination_loss
    )

    kl_parts = compute_factored_kl(
        posterior_q_mean=posterior.q_mean,
        posterior_q_std=posterior.q_std,
        posterior_qdot_mean=posterior.qdot_mean,
        posterior_qdot_std=posterior.qdot_std,
        posterior_z_mean=posterior.z_mean,
        posterior_z_std=posterior.z_std,
        prior_q_mean=rollout.prior_q_mean,
        prior_q_std=rollout.prior_q_std,
        prior_qdot_mean=rollout.prior_qdot_mean,
        prior_qdot_std=rollout.prior_qdot_std,
        prior_z_mean=rollout.prior_z_mean,
        prior_z_std=rollout.prior_z_std,
        kl_balance_alpha=kl_balance_alpha,
        kl_free_nats_mech=kl_free_nats_mech,
        kl_free_nats_nuisance=kl_free_nats_nuisance,
    )
    kl_loss = kl_parts["kl_loss"]
    weighted_kl_loss = kl_weight * kl_loss

    smoothness_loss = compute_qdot_smoothness_loss(
        q_mean=posterior.q_mean,
        qdot_mean=posterior.qdot_mean,
        dt=dt,
    )
    weighted_smoothness_loss = smoothness_weight * smoothness_loss

    reward_loss, imagined_reward_loss = _reward_losses(
        outputs=outputs,
        rewards=rewards,
        reward_weight=reward_weight,
        imagined_reward_weight=imagined_reward_weight,
        zero=zero,
    )
    weighted_reward_loss = reward_weight * reward_loss
    weighted_imagined_reward_loss = imagined_reward_weight * imagined_reward_loss

    total_loss = (
        weighted_reconstruction_loss
        + weighted_foreground_reconstruction_loss
        + weighted_imagination_loss
        + weighted_foreground_imagination_loss
        + weighted_kl_loss
        + weighted_smoothness_loss
        + weighted_reward_loss
        + weighted_imagined_reward_loss
    )

    losses: dict[str, torch.Tensor] = {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "weighted_reconstruction_loss": weighted_reconstruction_loss,
        "foreground_reconstruction_loss": foreground_reconstruction_loss,
        "weighted_foreground_reconstruction_loss": weighted_foreground_reconstruction_loss,
        "imagination_reconstruction_loss": imagination_loss,
        "weighted_imagination_reconstruction_loss": weighted_imagination_loss,
        "foreground_imagination_reconstruction_loss": foreground_imagination_loss,
        "weighted_foreground_imagination_reconstruction_loss": (
            weighted_foreground_imagination_loss
        ),
        "kl_loss": kl_loss,
        "weighted_kl_loss": weighted_kl_loss,
        "smoothness_loss": smoothness_loss,
        "weighted_smoothness_loss": weighted_smoothness_loss,
        "reward_loss": reward_loss,
        "weighted_reward_loss": weighted_reward_loss,
        "imagined_reward_loss": imagined_reward_loss,
        "weighted_imagined_reward_loss": weighted_imagined_reward_loss,
    }
    for key, value in kl_parts.items():
        if key == "kl_loss":
            continue
        losses[key] = value
    return losses


def compute_factored_kl(
    posterior_q_mean: torch.Tensor,
    posterior_q_std: torch.Tensor,
    posterior_qdot_mean: torch.Tensor,
    posterior_qdot_std: torch.Tensor,
    posterior_z_mean: torch.Tensor,
    posterior_z_std: torch.Tensor,
    prior_q_mean: torch.Tensor,
    prior_q_std: torch.Tensor,
    prior_qdot_mean: torch.Tensor,
    prior_qdot_std: torch.Tensor,
    prior_z_mean: torch.Tensor,
    prior_z_std: torch.Tensor,
    kl_balance_alpha: float,
    kl_free_nats_mech: float,
    kl_free_nats_nuisance: float,
) -> dict[str, torch.Tensor]:
    """Three-branch balanced KL with per-branch free-nats floors.

    Mechanics branches ``q`` and ``qdot`` share a tighter floor than the
    nuisance branch so the model can't hide information inside ``z`` while
    the mechanics prior stays clamped at zero gradient.
    """

    q_parts = _balanced_kl(
        posterior_q_mean,
        posterior_q_std,
        prior_q_mean,
        prior_q_std,
        kl_balance_alpha=kl_balance_alpha,
        free_nats=kl_free_nats_mech,
    )
    qdot_parts = _balanced_kl(
        posterior_qdot_mean,
        posterior_qdot_std,
        prior_qdot_mean,
        prior_qdot_std,
        kl_balance_alpha=kl_balance_alpha,
        free_nats=kl_free_nats_mech,
    )
    z_parts = _balanced_kl(
        posterior_z_mean,
        posterior_z_std,
        prior_z_mean,
        prior_z_std,
        kl_balance_alpha=kl_balance_alpha,
        free_nats=kl_free_nats_nuisance,
    )
    kl_loss = q_parts["kl_loss"] + qdot_parts["kl_loss"] + z_parts["kl_loss"]
    return {
        "kl_loss": kl_loss,
        "kl_q_loss": q_parts["kl_loss"],
        "kl_qdot_loss": qdot_parts["kl_loss"],
        "kl_z_loss": z_parts["kl_loss"],
        "kl_q_raw": q_parts["kl_raw"],
        "kl_qdot_raw": qdot_parts["kl_raw"],
        "kl_z_raw": z_parts["kl_raw"],
        "kl_q_free_nats_active": q_parts["kl_free_nats_active"],
        "kl_qdot_free_nats_active": qdot_parts["kl_free_nats_active"],
        "kl_z_free_nats_active": z_parts["kl_free_nats_active"],
    }


def compute_qdot_smoothness_loss(
    q_mean: torch.Tensor,
    qdot_mean: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Finite-difference smoothness prior on the encoder's qdot head.

    ``qdot_t`` should equal ``(q_{t+1} - q_t) / dt`` — the single most
    informative regularizer against state collapse, because it forces the
    encoder's two mechanics heads into a derivative relationship the
    Lagrangian step expects. The loss is averaged over ``B * (T-1) * d``.
    """

    if q_mean.shape != qdot_mean.shape:
        raise ValueError("q and qdot tensors must share shape")
    if q_mean.shape[1] < 2:
        return q_mean.new_zeros(())
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    finite_diff = (q_mean[:, 1:] - q_mean[:, :-1]) / dt
    midpoint = 0.5 * (qdot_mean[:, 1:] + qdot_mean[:, :-1])
    return F.mse_loss(midpoint, finite_diff.detach())


def _balanced_kl(
    posterior_mean: torch.Tensor,
    posterior_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    kl_balance_alpha: float,
    free_nats: float,
) -> dict[str, torch.Tensor]:
    """DreamerV2-style balanced KL for a single Gaussian branch."""

    if not 0.0 <= kl_balance_alpha <= 1.0:
        raise ValueError("kl_balance_alpha must satisfy 0 <= alpha <= 1")
    if free_nats < 0.0:
        raise ValueError("free_nats must be >= 0")

    kl_post_term = diagonal_gaussian_kl(
        mean_q=posterior_mean.detach(),
        std_q=posterior_std.detach(),
        mean_p=prior_mean,
        std_p=prior_std,
    )
    kl_prior_term = diagonal_gaussian_kl(
        mean_q=posterior_mean,
        std_q=posterior_std,
        mean_p=prior_mean.detach(),
        std_p=prior_std.detach(),
    )
    kl_balanced = kl_balance_alpha * kl_post_term + (1.0 - kl_balance_alpha) * kl_prior_term
    kl_clamped = torch.clamp(kl_balanced, min=free_nats)
    with torch.no_grad():
        kl_raw = diagonal_gaussian_kl(
            mean_q=posterior_mean,
            std_q=posterior_std,
            mean_p=prior_mean,
            std_p=prior_std,
        ).mean()
        free_nats_active = (kl_balanced.detach() <= free_nats).float().mean()
    return {
        "kl_loss": kl_clamped.mean(),
        "kl_raw": kl_raw,
        "kl_free_nats_active": free_nats_active,
    }


def _foreground_mse(
    reconstructions: torch.Tensor,
    observations: torch.Tensor,
    weight: float,
    floor: float,
    kernel_size: int,
    zero: torch.Tensor,
) -> torch.Tensor:
    if weight < 0.0:
        raise ValueError("foreground_reconstruction_weight must be >= 0")
    if weight == 0.0:
        return zero
    if reconstructions.shape != observations.shape:
        raise ValueError(
            f"reconstruction shape mismatch: {reconstructions.shape} != {observations.shape}",
        )
    mask = foreground_reconstruction_mask(
        observations=observations,
        floor=floor,
        kernel_size=kernel_size,
    )
    squared_error = (reconstructions - observations).pow(2)
    return (squared_error * mask.detach()).mean()


def _imagination_mse(
    imagined: torch.Tensor | None,
    target_start: int | None,
    next_observations: torch.Tensor | None,
    weight: float,
    fg_weight: float,
    foreground_mask_floor: float,
    foreground_mask_kernel_size: int,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight < 0.0:
        raise ValueError("imagination_reconstruction_weight must be >= 0")
    if fg_weight < 0.0:
        raise ValueError("foreground_imagination_reconstruction_weight must be >= 0")
    if weight == 0.0 and fg_weight == 0.0:
        return zero, zero
    if imagined is None or target_start is None:
        raise ValueError(
            "imagination_context_steps and imagination_horizon must be > 0 when "
            "imagination reconstruction weights are > 0",
        )
    if next_observations is None:
        raise ValueError(
            "next_observations must be passed when imagination reconstruction "
            "weights are > 0",
        )
    horizon = imagined.shape[1]
    if next_observations.shape[1] < target_start + horizon:
        raise ValueError(
            "next_observations too short for imagination targets: need length "
            f"{target_start + horizon}, got {next_observations.shape[1]}",
        )
    target = next_observations[:, target_start : target_start + horizon]
    if imagined.shape != target.shape:
        raise ValueError(
            f"imagination target shape mismatch: {imagined.shape} vs {target.shape}",
        )
    squared_error = (imagined - target).pow(2)
    imagination_loss = squared_error.mean() if weight > 0.0 else zero
    if fg_weight > 0.0:
        mask = foreground_reconstruction_mask(
            observations=target,
            floor=foreground_mask_floor,
            kernel_size=foreground_mask_kernel_size,
        )
        fg_loss = (squared_error * mask.detach()).mean()
    else:
        fg_loss = zero
    return imagination_loss, fg_loss


def _reward_losses(
    outputs: MechanicsWorldModelOutput,
    rewards: torch.Tensor | None,
    reward_weight: float,
    imagined_reward_weight: float,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reward_weight < 0.0:
        raise ValueError("reward_weight must be >= 0")
    if imagined_reward_weight < 0.0:
        raise ValueError("imagined_reward_weight must be >= 0")
    if reward_weight == 0.0 and imagined_reward_weight == 0.0:
        return zero, zero
    if outputs.reward_predictions is None:
        raise ValueError("reward_head=True required on the model when reward_weight > 0")
    if rewards is None:
        raise ValueError("rewards tensor must be passed when reward weights are > 0")
    if outputs.reward_predictions.shape != rewards.shape:
        raise ValueError(
            f"reward shape mismatch: {outputs.reward_predictions.shape} vs {rewards.shape}",
        )
    posterior_loss = (
        F.mse_loss(outputs.reward_predictions, rewards) if reward_weight > 0.0 else zero
    )
    if imagined_reward_weight > 0.0:
        if outputs.imagined_reward_predictions is None or outputs.imagined_target_start is None:
            raise ValueError(
                "imagination_horizon must be > 0 on the model when "
                "imagined_reward_weight > 0",
            )
        horizon = outputs.imagined_reward_predictions.shape[1]
        target_start = outputs.imagined_target_start
        if rewards.shape[1] < target_start + horizon:
            raise ValueError(
                "rewards too short for imagination targets: need length "
                f"{target_start + horizon}, got {rewards.shape[1]}",
            )
        target = rewards[:, target_start : target_start + horizon]
        imagined_loss = F.mse_loss(outputs.imagined_reward_predictions, target)
    else:
        imagined_loss = zero
    return posterior_loss, imagined_loss
