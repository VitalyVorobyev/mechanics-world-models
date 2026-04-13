from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models import (
    ConvEncoder,
    ImageDecoder,
    RSSM,
    VisualWorldModel,
    compute_imagination_reconstruction_losses,
    compute_latent_consistency_loss,
    compute_rssm_kl_loss,
    compute_world_model_loss,
    foreground_reconstruction_mask,
)
from train.train_rssm import (
    build_arg_parser,
    build_lr_scheduler,
    capture_rng_state,
    load_checkpoint,
    parse_horizons_arg,
    restore_rng_state,
    run_val_open_loop_probe,
    save_checkpoint,
    serializable_config,
    validate_resume_config,
)
from train.train_rssm import TrainConfig


def _observations(batch_size: int = 2, sequence_length: int = 3) -> torch.Tensor:
    return torch.rand(batch_size, sequence_length, 3, 84, 84)


def _actions(batch_size: int = 2, sequence_length: int = 3, action_dim: int = 2) -> torch.Tensor:
    return torch.rand(batch_size, sequence_length, action_dim)


def test_encoder_output_shape() -> None:
    encoder = ConvEncoder(embedding_size=32)
    embeddings = encoder(_observations())

    assert embeddings.shape == (2, 3, 32)
    assert embeddings.dtype == torch.float32


def test_rssm_rollout_output_shapes() -> None:
    rssm = RSSM(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    outputs = rssm(embeddings=torch.rand(2, 3, 32), actions=_actions())

    assert outputs.prior_mean.shape == (2, 3, 8)
    assert outputs.prior_std.shape == (2, 3, 8)
    assert outputs.posterior_mean.shape == (2, 3, 8)
    assert outputs.posterior_std.shape == (2, 3, 8)
    assert outputs.latents.shape == (2, 3, 8)
    assert outputs.deter_states.shape == (2, 3, 16)
    assert outputs.features.shape == (2, 3, 24)
    assert torch.allclose(outputs.deter_states[:, 0], torch.zeros_like(outputs.deter_states[:, 0]))
    assert torch.all(outputs.prior_std > 0.0)
    assert torch.all(outputs.posterior_std > 0.0)


def test_decoder_output_shape() -> None:
    decoder = ImageDecoder(feature_size=24, output_size=84)
    reconstructions = decoder(torch.rand(2, 3, 24))

    assert reconstructions.shape == (2, 3, 3, 84, 84)
    assert torch.all(reconstructions >= 0.0)
    assert torch.all(reconstructions <= 1.0)


def test_world_model_forward_shape_consistency() -> None:
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    outputs = model(observations=_observations(), actions=_actions())

    assert outputs.embeddings.shape == (2, 3, 32)
    assert outputs.rssm.features.shape == (2, 3, 24)
    assert outputs.reconstructions.shape == (2, 3, 3, 84, 84)


def test_world_model_forward_with_next_observations_returns_latent_predictions() -> None:
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    outputs = model(
        observations=_observations(),
        actions=_actions(),
        next_observations=_observations(),
    )

    assert outputs.next_embeddings is not None
    assert outputs.predicted_next_embeddings is not None
    assert outputs.next_prior_features is not None
    assert outputs.next_reconstructions is not None
    assert outputs.next_embeddings.shape == (2, 3, 32)
    assert outputs.predicted_next_embeddings.shape == (2, 3, 32)
    assert outputs.next_prior_features.shape == (2, 3, 24)
    assert outputs.next_reconstructions.shape == (2, 3, 3, 84, 84)


def test_transition_aligned_next_prior_features_shape() -> None:
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    features = model.transition_aligned_next_prior_features(
        embeddings=torch.rand(2, 3, 32),
        actions=_actions(),
    )

    assert features.shape == (2, 3, 24)


def test_world_model_loss_returns_finite_values() -> None:
    observations = _observations()
    actions = _actions()
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)

    outputs = model(observations=observations, actions=actions)
    losses = compute_world_model_loss(outputs, observations, kl_weight=0.1)

    assert set(losses) == {
        "total_loss",
        "reconstruction_loss",
        "weighted_reconstruction_loss",
        "foreground_reconstruction_loss",
        "weighted_foreground_reconstruction_loss",
        "transition_reconstruction_loss",
        "weighted_transition_reconstruction_loss",
        "foreground_transition_reconstruction_loss",
        "weighted_foreground_transition_reconstruction_loss",
        "imagination_reconstruction_loss",
        "weighted_imagination_reconstruction_loss",
        "foreground_imagination_reconstruction_loss",
        "weighted_foreground_imagination_reconstruction_loss",
        "dynamic_reconstruction_loss",
        "weighted_dynamic_reconstruction_loss",
        "kl_loss",
        "weighted_kl_loss",
        "kl_raw",
        "kl_balanced_raw",
        "kl_free_nats_active",
        "latent_consistency_loss",
        "weighted_latent_consistency_loss",
        "reward_loss",
        "weighted_reward_loss",
        "imagined_reward_loss",
        "weighted_imagined_reward_loss",
    }
    assert torch.allclose(losses["weighted_reconstruction_loss"], losses["reconstruction_loss"])
    assert torch.allclose(
        losses["foreground_reconstruction_loss"],
        torch.zeros_like(losses["foreground_reconstruction_loss"]),
    )
    assert torch.allclose(
        losses["dynamic_reconstruction_loss"],
        torch.zeros_like(losses["dynamic_reconstruction_loss"]),
    )
    assert torch.allclose(
        losses["transition_reconstruction_loss"],
        torch.zeros_like(losses["transition_reconstruction_loss"]),
    )
    assert torch.allclose(
        losses["foreground_transition_reconstruction_loss"],
        torch.zeros_like(losses["foreground_transition_reconstruction_loss"]),
    )
    assert torch.allclose(losses["weighted_kl_loss"], 0.1 * losses["kl_loss"])
    assert torch.allclose(
        losses["latent_consistency_loss"],
        torch.zeros_like(losses["latent_consistency_loss"]),
    )
    assert torch.allclose(
        losses["total_loss"],
        losses["reconstruction_loss"] + losses["weighted_kl_loss"],
    )
    for loss in losses.values():
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_latent_consistency_loss_returns_finite_value() -> None:
    observations = _observations()
    next_observations = _observations()
    actions = _actions()
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)

    outputs = model(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
    )
    latent_loss = compute_latent_consistency_loss(outputs, latent_consistency_weight=1.0)
    losses = compute_world_model_loss(
        outputs,
        observations,
        kl_weight=0.1,
        reconstruction_weight=0.05,
        latent_consistency_weight=1.0,
    )

    assert latent_loss.ndim == 0
    assert torch.isfinite(latent_loss)
    assert torch.allclose(losses["latent_consistency_loss"], latent_loss)
    assert torch.allclose(
        losses["weighted_reconstruction_loss"],
        0.05 * losses["reconstruction_loss"],
    )
    assert torch.allclose(
        losses["total_loss"],
        losses["weighted_reconstruction_loss"]
        + losses["weighted_kl_loss"]
        + losses["weighted_latent_consistency_loss"],
    )


def test_foreground_reconstruction_mask_shape_and_normalization() -> None:
    observations = _observations()

    mask = foreground_reconstruction_mask(observations, floor=0.02, kernel_size=7)

    assert mask.shape == (2, 3, 1, 84, 84)
    assert mask.dtype == observations.dtype
    assert torch.all(torch.isfinite(mask))
    per_frame_mean = mask.mean(dim=(-2, -1))
    assert torch.allclose(per_frame_mean, torch.ones_like(per_frame_mean), atol=1e-5)


def test_foreground_reconstruction_loss_returns_finite_value() -> None:
    observations = _observations()
    actions = _actions()
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)

    outputs = model(observations=observations, actions=actions)
    losses = compute_world_model_loss(
        outputs,
        observations,
        kl_weight=0.1,
        reconstruction_weight=0.05,
        foreground_reconstruction_weight=2.0,
        foreground_mask_floor=0.02,
        foreground_mask_kernel_size=7,
    )

    assert losses["foreground_reconstruction_loss"].ndim == 0
    assert torch.isfinite(losses["foreground_reconstruction_loss"])
    assert torch.allclose(
        losses["weighted_foreground_reconstruction_loss"],
        2.0 * losses["foreground_reconstruction_loss"],
    )
    assert torch.allclose(
        losses["total_loss"],
        losses["weighted_reconstruction_loss"]
        + losses["weighted_foreground_reconstruction_loss"]
        + losses["weighted_dynamic_reconstruction_loss"]
        + losses["weighted_kl_loss"]
        + losses["weighted_latent_consistency_loss"],
    )


def test_dynamic_reconstruction_loss_returns_finite_value() -> None:
    observations = _observations()
    next_observations = observations.clone()
    next_observations[:, :, :, 30:54, 30:54] = 1.0 - next_observations[:, :, :, 30:54, 30:54]
    actions = _actions()
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)

    outputs = model(observations=observations, actions=actions)
    losses = compute_world_model_loss(
        outputs,
        observations,
        kl_weight=0.1,
        reconstruction_weight=0.05,
        dynamic_reconstruction_weight=1.0,
        next_observations=next_observations,
    )

    assert losses["dynamic_reconstruction_loss"].ndim == 0
    assert torch.isfinite(losses["dynamic_reconstruction_loss"])
    assert torch.allclose(
        losses["weighted_dynamic_reconstruction_loss"],
        losses["dynamic_reconstruction_loss"],
    )
    assert torch.allclose(
        losses["total_loss"],
        losses["weighted_reconstruction_loss"]
        + losses["weighted_dynamic_reconstruction_loss"]
        + losses["weighted_kl_loss"]
        + losses["weighted_latent_consistency_loss"],
    )


def test_dynamic_reconstruction_requires_next_observations_when_weighted() -> None:
    observations = _observations()
    actions = _actions()
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    outputs = model(observations=observations, actions=actions)

    with pytest.raises(ValueError, match="next_observations"):
        compute_world_model_loss(
            outputs,
            observations,
            kl_weight=0.1,
            dynamic_reconstruction_weight=1.0,
        )


def test_transition_reconstruction_loss_returns_finite_value() -> None:
    observations = _observations()
    next_observations = _observations()
    actions = _actions()
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)

    outputs = model(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
    )
    losses = compute_world_model_loss(
        outputs,
        observations,
        kl_weight=0.1,
        reconstruction_weight=0.05,
        transition_reconstruction_weight=1.5,
        next_observations=next_observations,
    )

    assert losses["transition_reconstruction_loss"].ndim == 0
    assert torch.isfinite(losses["transition_reconstruction_loss"])
    assert torch.allclose(
        losses["weighted_transition_reconstruction_loss"],
        1.5 * losses["transition_reconstruction_loss"],
    )
    assert torch.allclose(
        losses["total_loss"],
        losses["weighted_reconstruction_loss"]
        + losses["weighted_dynamic_reconstruction_loss"]
        + losses["weighted_foreground_reconstruction_loss"]
        + losses["weighted_transition_reconstruction_loss"]
        + losses["weighted_foreground_transition_reconstruction_loss"]
        + losses["weighted_kl_loss"]
        + losses["weighted_latent_consistency_loss"],
    )


def test_foreground_transition_reconstruction_loss_returns_finite_value() -> None:
    observations = _observations()
    next_observations = _observations()
    actions = _actions()
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)

    outputs = model(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
    )
    losses = compute_world_model_loss(
        outputs,
        observations,
        kl_weight=0.1,
        reconstruction_weight=0.05,
        foreground_transition_reconstruction_weight=2.0,
        foreground_mask_floor=0.02,
        foreground_mask_kernel_size=7,
        next_observations=next_observations,
    )

    assert losses["foreground_transition_reconstruction_loss"].ndim == 0
    assert torch.isfinite(losses["foreground_transition_reconstruction_loss"])
    assert torch.allclose(
        losses["weighted_foreground_transition_reconstruction_loss"],
        2.0 * losses["foreground_transition_reconstruction_loss"],
    )
    assert torch.allclose(
        losses["total_loss"],
        losses["weighted_reconstruction_loss"]
        + losses["weighted_dynamic_reconstruction_loss"]
        + losses["weighted_foreground_reconstruction_loss"]
        + losses["weighted_transition_reconstruction_loss"]
        + losses["weighted_foreground_transition_reconstruction_loss"]
        + losses["weighted_kl_loss"]
        + losses["weighted_latent_consistency_loss"],
    )


def test_transition_reconstruction_requires_next_outputs_when_weighted() -> None:
    observations = _observations()
    next_observations = _observations()
    actions = _actions()
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    outputs_without_next = model(observations=observations, actions=actions)
    outputs_with_next = model(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
    )

    with pytest.raises(ValueError, match="next_observations"):
        compute_world_model_loss(
            outputs_without_next,
            observations,
            kl_weight=0.1,
            transition_reconstruction_weight=1.0,
            next_observations=next_observations,
        )
    with pytest.raises(ValueError, match="next_observations"):
        compute_world_model_loss(
            outputs_with_next,
            observations,
            kl_weight=0.1,
            transition_reconstruction_weight=1.0,
        )


def test_latent_consistency_requires_next_observations_when_weighted() -> None:
    observations = _observations()
    actions = _actions()
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    outputs = model(observations=observations, actions=actions)

    with pytest.raises(ValueError, match="next_observations"):
        compute_world_model_loss(
            outputs,
            observations,
            kl_weight=0.1,
            latent_consistency_weight=1.0,
        )


def test_short_training_step_runs_without_nans() -> None:
    torch.manual_seed(0)
    observations = _observations()
    next_observations = _observations()
    actions = _actions()
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    outputs = model(observations=observations, actions=actions, next_observations=next_observations)
    losses = compute_world_model_loss(
        outputs,
        observations,
        kl_weight=0.1,
        reconstruction_weight=0.05,
        foreground_reconstruction_weight=2.0,
        foreground_transition_reconstruction_weight=1.0,
        dynamic_reconstruction_weight=1.0,
        next_observations=next_observations,
        latent_consistency_weight=1.0,
    )
    optimizer.zero_grad(set_to_none=True)
    losses["total_loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    assert torch.isfinite(losses["total_loss"])
    for parameter in model.parameters():
        assert torch.all(torch.isfinite(parameter))


def test_train_config_exposes_latent_consistency_options() -> None:
    option_dests = {action.dest for action in build_arg_parser()._actions}
    config = TrainConfig(
        dataset_dir=Path("data"),
        checkpoint_dir=Path("checkpoints"),
        reconstruction_weight=0.05,
        foreground_reconstruction_weight=2.0,
        foreground_mask_floor=0.02,
        foreground_mask_kernel_size=7,
        transition_reconstruction_weight=1.5,
        foreground_transition_reconstruction_weight=2.0,
        dynamic_reconstruction_weight=1.0,
        dynamic_mask_floor=0.1,
        latent_consistency_weight=1.0,
        kl_balance_alpha=0.7,
        kl_free_nats=2.5,
    )
    serialized = serializable_config(config)

    assert "reconstruction_weight" in option_dests
    assert "foreground_reconstruction_weight" in option_dests
    assert "foreground_mask_floor" in option_dests
    assert "foreground_mask_kernel_size" in option_dests
    assert "transition_reconstruction_weight" in option_dests
    assert "foreground_transition_reconstruction_weight" in option_dests
    assert "dynamic_reconstruction_weight" in option_dests
    assert "dynamic_mask_floor" in option_dests
    assert "latent_consistency_weight" in option_dests
    assert "kl_balance_alpha" in option_dests
    assert "kl_free_nats" in option_dests
    assert serialized["reconstruction_weight"] == 0.05
    assert serialized["foreground_reconstruction_weight"] == 2.0
    assert serialized["foreground_mask_floor"] == 0.02
    assert serialized["foreground_mask_kernel_size"] == 7
    assert serialized["transition_reconstruction_weight"] == 1.5
    assert serialized["foreground_transition_reconstruction_weight"] == 2.0
    assert serialized["dynamic_reconstruction_weight"] == 1.0
    assert serialized["dynamic_mask_floor"] == 0.1
    assert serialized["latent_consistency_weight"] == 1.0
    assert serialized["kl_balance_alpha"] == 0.7
    assert serialized["kl_free_nats"] == 2.5


def test_kl_balancing_matches_closed_form() -> None:
    torch.manual_seed(0)
    posterior_mean = torch.randn(2, 3, 8, requires_grad=True)
    posterior_std = (torch.rand(2, 3, 8) + 0.2).detach().requires_grad_(True)
    prior_mean = torch.randn(2, 3, 8, requires_grad=True)
    prior_std = (torch.rand(2, 3, 8) + 0.2).detach().requires_grad_(True)

    alpha = 0.8
    parts = compute_rssm_kl_loss(
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        prior_mean=prior_mean,
        prior_std=prior_std,
        kl_balance_alpha=alpha,
        kl_free_nats=0.0,
    )

    # Closed-form reproduction of the DreamerV2 balanced KL with no free-nats.
    def kl(mean_q, std_q, mean_p, std_p):
        var_q, var_p = std_q.pow(2), std_p.pow(2)
        per_dim = (
            torch.log(std_p / std_q)
            + (var_q + (mean_q - mean_p).pow(2)) / (2.0 * var_p)
            - 0.5
        )
        return per_dim.sum(dim=-1)

    expected_post = kl(posterior_mean.detach(), posterior_std.detach(), prior_mean, prior_std)
    expected_prior = kl(posterior_mean, posterior_std, prior_mean.detach(), prior_std.detach())
    expected = (alpha * expected_post + (1.0 - alpha) * expected_prior).mean()

    assert torch.allclose(parts["kl_loss"], expected, atol=1e-6)
    # kl_raw is the unbalanced KL(q||p) mean; it must be finite and positive.
    assert parts["kl_raw"].item() > 0.0


def test_kl_free_nats_floor_binds_when_kl_is_small() -> None:
    torch.manual_seed(0)
    # Posterior and prior are nearly identical, so KL is tiny and the floor binds.
    shape = (2, 3, 8)
    posterior_mean = torch.zeros(shape, requires_grad=True)
    posterior_std = torch.ones(shape, requires_grad=True)
    prior_mean = torch.zeros(shape, requires_grad=True)
    prior_std = torch.ones(shape, requires_grad=True)

    free_nats = 3.0
    parts = compute_rssm_kl_loss(
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        prior_mean=prior_mean,
        prior_std=prior_std,
        kl_balance_alpha=0.8,
        kl_free_nats=free_nats,
    )

    assert parts["kl_balanced_raw"].item() < free_nats
    assert torch.isclose(parts["kl_loss"], torch.tensor(free_nats))
    assert torch.isclose(parts["kl_free_nats_active"], torch.tensor(1.0))

    # Under the free-nats floor the balanced KL should contribute no gradient.
    parts["kl_loss"].backward()
    assert posterior_mean.grad is None or torch.allclose(
        posterior_mean.grad, torch.zeros_like(posterior_mean)
    )
    assert prior_mean.grad is None or torch.allclose(
        prior_mean.grad, torch.zeros_like(prior_mean)
    )


def test_imagination_rollout_shape_and_target_alignment() -> None:
    batch_size, sequence_length, action_dim = 2, 10, 2
    context_steps, horizon = 3, 5
    observations = torch.rand(batch_size, sequence_length, 3, 84, 84)
    next_observations = torch.rand(batch_size, sequence_length, 3, 84, 84)
    actions = torch.rand(batch_size, sequence_length, action_dim)
    model = VisualWorldModel(
        action_dim=action_dim,
        embedding_size=32,
        hidden_size=16,
        latent_size=8,
    )

    outputs = model(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        imagination_context_steps=context_steps,
        imagination_horizon=horizon,
    )

    assert outputs.imagined_reconstructions is not None
    assert outputs.imagined_features is not None
    assert outputs.imagined_reconstructions.shape == (batch_size, horizon, 3, 84, 84)
    assert outputs.imagined_features.shape == (batch_size, horizon, 16 + 8)
    assert outputs.imagined_target_start == context_steps - 1
    assert torch.all(outputs.imagined_reconstructions >= 0.0)
    assert torch.all(outputs.imagined_reconstructions <= 1.0)


def test_imagination_loss_targets_correct_next_observation_slice() -> None:
    torch.manual_seed(0)
    batch_size, sequence_length, action_dim = 2, 10, 2
    context_steps, horizon = 3, 5
    target_start = context_steps - 1  # index 2 ─ obs_3 is next_obs[:, 2]
    observations = torch.rand(batch_size, sequence_length, 3, 84, 84)
    next_observations = torch.rand(batch_size, sequence_length, 3, 84, 84)
    actions = torch.rand(batch_size, sequence_length, action_dim)
    model = VisualWorldModel(
        action_dim=action_dim,
        embedding_size=32,
        hidden_size=16,
        latent_size=8,
    )
    outputs = model(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        imagination_context_steps=context_steps,
        imagination_horizon=horizon,
    )

    imagination_loss, foreground_imagination_loss = compute_imagination_reconstruction_losses(
        outputs=outputs,
        next_observations=next_observations,
        imagination_reconstruction_weight=1.0,
        foreground_imagination_reconstruction_weight=2.0,
        foreground_mask_floor=0.02,
        foreground_mask_kernel_size=7,
    )

    expected_target = next_observations[:, target_start : target_start + horizon]
    expected_loss = (outputs.imagined_reconstructions - expected_target).pow(2).mean()
    assert torch.allclose(imagination_loss, expected_loss, atol=1e-6)
    assert torch.isfinite(foreground_imagination_loss)
    assert foreground_imagination_loss.item() >= 0.0


def test_imagination_gradients_reach_rssm_and_decoder() -> None:
    torch.manual_seed(0)
    observations = torch.rand(2, 10, 3, 84, 84)
    next_observations = torch.rand(2, 10, 3, 84, 84)
    actions = torch.rand(2, 10, 2)
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)

    outputs = model(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        imagination_context_steps=3,
        imagination_horizon=5,
    )
    losses = compute_world_model_loss(
        outputs,
        observations,
        kl_weight=0.0,
        reconstruction_weight=0.0,
        foreground_imagination_reconstruction_weight=2.0,
        next_observations=next_observations,
    )
    losses["total_loss"].backward()

    decoder_grad = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in model.decoder.parameters()
    )
    recurrent_grad = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in model.rssm.recurrent.parameters()
    )
    prior_grad = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in model.rssm.prior.parameters()
    )
    encoder_grad = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in model.encoder.parameters()
    )
    assert decoder_grad > 0.0
    assert recurrent_grad > 0.0
    assert prior_grad > 0.0
    # Encoder only contributes via the posterior context used to seed the rollout.
    assert encoder_grad > 0.0


def test_imagination_raises_on_short_sequence() -> None:
    # context=4 + horizon=12 ⇒ need T >= 15; T=10 is too short.
    observations = torch.rand(2, 10, 3, 84, 84)
    actions = torch.rand(2, 10, 2)
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    with pytest.raises(ValueError, match="imagination requires sequence_length"):
        model(
            observations=observations,
            actions=actions,
            imagination_context_steps=4,
            imagination_horizon=12,
        )


def test_imagination_loss_requires_model_imagination_outputs() -> None:
    observations = torch.rand(2, 10, 3, 84, 84)
    next_observations = torch.rand(2, 10, 3, 84, 84)
    actions = torch.rand(2, 10, 2)
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    outputs_no_imagination = model(observations=observations, actions=actions)
    with pytest.raises(ValueError, match="imagination_context_steps"):
        compute_world_model_loss(
            outputs_no_imagination,
            observations,
            kl_weight=0.0,
            imagination_reconstruction_weight=1.0,
            next_observations=next_observations,
        )


def test_warmup_cosine_scheduler_lr_profile() -> None:
    # LambdaLR's construction advances last_epoch from -1 to 0, so the lambda
    # is first called at step 0 during __init__; each subsequent step advances
    # last_epoch by 1. Peak LR is reached at last_epoch == warmup_steps - 1.
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    base_lr = 3e-4
    warmup = 4
    total = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=warmup,
        total_steps=total,
        schedule="cosine",
    )

    # lrs[step] = LR used for optimizer step `step` (zero-indexed), matching
    # how scheduler.step() is called *after* optimizer.step() in training.
    lrs = [scheduler.get_last_lr()[0]]
    for _ in range(total - 1):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    # Warmup occupies indices 0..warmup-1 with strictly increasing LR, ending at peak.
    for prev, cur in zip(lrs[: warmup - 1], lrs[1:warmup]):
        assert cur > prev
    assert abs(lrs[warmup - 1] - base_lr) < 1e-7
    # After warmup, cosine decays monotonically to 0 at the last step.
    for prev, cur in zip(lrs[warmup - 1 :], lrs[warmup:]):
        assert cur <= prev + 1e-9
    assert lrs[-1] < 1e-9


def test_parse_horizons_arg_sorts_and_deduplicates() -> None:
    assert parse_horizons_arg("1,5,10") == (1, 5, 10)
    assert parse_horizons_arg("10, 5, 5, 1") == (1, 5, 10)
    with pytest.raises(ValueError, match="positive"):
        parse_horizons_arg("0,5,10")


def test_val_open_loop_probe_produces_per_horizon_fg_mse() -> None:
    torch.manual_seed(0)
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    sequence_length = 25
    batch = {
        "observations": torch.rand(4, sequence_length, 3, 84, 84),
        "actions": torch.rand(4, sequence_length, 2),
        "next_observations": torch.rand(4, sequence_length, 3, 84, 84),
    }

    model.train()  # probe must restore training mode afterwards
    metrics = run_val_open_loop_probe(
        model=model,
        batch=batch,
        warmup=4,
        horizons=(1, 5, 10, 20),
        foreground_mask_floor=0.02,
        foreground_mask_kernel_size=7,
    )

    assert model.training  # restored
    for h in (1, 5, 10, 20):
        assert f"val_open_loop_mse_h{h}" in metrics
        assert f"val_open_loop_fg_mse_h{h}" in metrics
        assert metrics[f"val_open_loop_mse_h{h}"] >= 0.0
        assert metrics[f"val_open_loop_fg_mse_h{h}"] >= 0.0


def test_val_open_loop_probe_raises_on_short_sequence() -> None:
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    batch = {
        "observations": torch.rand(2, 10, 3, 84, 84),
        "actions": torch.rand(2, 10, 2),
        "next_observations": torch.rand(2, 10, 3, 84, 84),
    }
    with pytest.raises(ValueError, match="val_open_loop probe"):
        run_val_open_loop_probe(
            model=model,
            batch=batch,
            warmup=4,
            horizons=(1, 5, 10, 20),
            foreground_mask_floor=0.02,
            foreground_mask_kernel_size=7,
        )


def test_checkpoint_round_trip_restores_state(tmp_path: Path) -> None:
    config = TrainConfig(
        dataset_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "ckpt",
        sequence_length=16,
        hidden_size=16,
        latent_size=8,
        embedding_size=32,
    )
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = build_lr_scheduler(optimizer, warmup_steps=2, total_steps=10, schedule="cosine")
    config.checkpoint_dir.mkdir(parents=True)

    # Burn one optimizer step so the scheduler advances and Adam state has buffers.
    loss = sum(p.sum() for p in model.parameters())
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_lr = scheduler.get_last_lr()[0]

    saved_path = save_checkpoint(
        checkpoint_dir=config.checkpoint_dir,
        epoch=3,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        metrics={"total_loss": 1.23},
        global_step=42,
        history_path=config.checkpoint_dir / "history.jsonl",
    )

    fresh_model = VisualWorldModel(
        action_dim=2, embedding_size=32, hidden_size=16, latent_size=8,
    )
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=3e-4)
    fresh_scheduler = build_lr_scheduler(
        fresh_optimizer, warmup_steps=2, total_steps=10, schedule="cosine",
    )
    epoch, global_step = load_checkpoint(
        saved_path,
        fresh_model,
        fresh_optimizer,
        device=torch.device("cpu"),
        scheduler=fresh_scheduler,
        expected_config=config,
    )

    assert epoch == 3
    assert global_step == 42
    # Scheduler resumed at the correct LR step.
    assert abs(fresh_scheduler.get_last_lr()[0] - expected_lr) < 1e-9
    # Model weights are bit-identical.
    for original, restored in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(original.detach(), restored.detach())


def test_validate_resume_config_blocks_architecture_drift() -> None:
    base = serializable_config(
        TrainConfig(dataset_dir=Path("a"), checkpoint_dir=Path("b"), latent_size=8)
    )
    expected = TrainConfig(dataset_dir=Path("a"), checkpoint_dir=Path("b"), latent_size=16)
    with pytest.raises(RuntimeError, match="latent_size"):
        validate_resume_config(base, expected)


def test_validate_resume_config_allows_optimizer_knob_changes() -> None:
    base = serializable_config(
        TrainConfig(
            dataset_dir=Path("a"),
            checkpoint_dir=Path("b"),
            kl_weight=1.0,
            learning_rate=3e-4,
        )
    )
    expected = TrainConfig(
        dataset_dir=Path("a"),
        checkpoint_dir=Path("b"),
        kl_weight=0.5,
        learning_rate=1e-4,
    )
    validate_resume_config(base, expected)  # must not raise


def test_capture_and_restore_rng_state_round_trip() -> None:
    torch.manual_seed(123)
    state = capture_rng_state()
    drawn_first = torch.randn(3)

    # Different RNG state in between.
    torch.manual_seed(456)
    _ = torch.randn(3)

    restore_rng_state(state)
    drawn_after = torch.randn(3)
    assert torch.equal(drawn_first, drawn_after)


def test_restore_rng_state_normalizes_non_byte_dtype() -> None:
    """Regression for the ``map_location`` bug: on resume through a non-CPU
    device, ``torch.load`` relocates the RNG ByteTensor into non-CPU memory
    and/or a non-uint8 dtype, which ``torch.set_rng_state`` rejects. The
    restore helper must coerce back to CPU/uint8 without raising.
    """

    torch.manual_seed(7)
    state = capture_rng_state()
    # Simulate the post-load shape: same byte payload, different dtype.
    state["torch_cpu"] = state["torch_cpu"].to(dtype=torch.int64).to(dtype=torch.uint8)
    restore_rng_state(state)  # must not raise


def test_restore_rng_state_round_trip_through_torch_save_load(tmp_path: Path) -> None:
    """End-to-end check of the exact save → torch.load path that broke resume.

    ``torch.load(map_location="cpu")`` is the tamest codepath, but it still
    goes through ``_rebuild_tensor_v2`` on every tensor, which is what
    clobbered the dtype in production. This guards against any future
    regression in that path.
    """

    torch.manual_seed(1)
    payload = {"rng_state": capture_rng_state()}
    path = tmp_path / "rng.pt"
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restore_rng_state(loaded["rng_state"])  # must not raise


def test_reward_head_emits_per_step_predictions() -> None:
    model = VisualWorldModel(
        action_dim=2,
        embedding_size=32,
        hidden_size=16,
        latent_size=8,
        reward_head=True,
    )
    outputs = model(observations=_observations(), actions=_actions())

    assert outputs.reward_predictions is not None
    assert outputs.reward_predictions.shape == (2, 3)


def test_reward_head_imagined_predictions_match_horizon() -> None:
    model = VisualWorldModel(
        action_dim=2,
        embedding_size=32,
        hidden_size=16,
        latent_size=8,
        reward_head=True,
    )
    obs = torch.rand(2, 10, 3, 84, 84)
    next_obs = torch.rand(2, 10, 3, 84, 84)
    actions = torch.rand(2, 10, 2)
    outputs = model(
        observations=obs,
        actions=actions,
        next_observations=next_obs,
        imagination_context_steps=3,
        imagination_horizon=5,
    )

    assert outputs.imagined_reward_predictions is not None
    assert outputs.imagined_reward_predictions.shape == (2, 5)


def test_reward_loss_targets_correct_reward_slice() -> None:
    model = VisualWorldModel(
        action_dim=2,
        embedding_size=32,
        hidden_size=16,
        latent_size=8,
        reward_head=True,
    )
    obs = torch.rand(2, 10, 3, 84, 84)
    next_obs = torch.rand(2, 10, 3, 84, 84)
    actions = torch.rand(2, 10, 2)
    rewards = torch.rand(2, 10)
    outputs = model(
        observations=obs,
        actions=actions,
        next_observations=next_obs,
        imagination_context_steps=3,
        imagination_horizon=5,
    )
    losses = compute_world_model_loss(
        outputs,
        obs,
        kl_weight=0.0,
        reconstruction_weight=0.0,
        reward_weight=1.0,
        imagined_reward_weight=2.0,
        rewards=rewards,
        next_observations=next_obs,
    )

    expected_posterior = torch.nn.functional.mse_loss(outputs.reward_predictions, rewards)
    expected_imagined = torch.nn.functional.mse_loss(
        outputs.imagined_reward_predictions, rewards[:, 2:7]
    )
    assert torch.allclose(losses["reward_loss"], expected_posterior, atol=1e-6)
    assert torch.allclose(losses["imagined_reward_loss"], expected_imagined, atol=1e-6)
    assert torch.allclose(
        losses["weighted_reward_loss"] + losses["weighted_imagined_reward_loss"],
        losses["total_loss"],
        atol=1e-6,
    )


def test_reward_loss_requires_reward_head() -> None:
    model = VisualWorldModel(
        action_dim=2, embedding_size=32, hidden_size=16, latent_size=8, reward_head=False,
    )
    outputs = model(observations=_observations(), actions=_actions())
    with pytest.raises(ValueError, match="reward_head=True"):
        compute_world_model_loss(
            outputs,
            _observations(),
            kl_weight=0.0,
            reward_weight=1.0,
            rewards=torch.zeros(2, 3),
        )


def test_checkpoint_compat_without_to_with_reward_head(tmp_path: Path) -> None:
    """Checkpoint trained without reward head loads into a model with reward head."""

    config = TrainConfig(
        dataset_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "ckpt",
        sequence_length=16,
        hidden_size=16,
        latent_size=8,
        embedding_size=32,
        reward_head=False,
    )
    config.checkpoint_dir.mkdir(parents=True)
    plain_model = VisualWorldModel(
        action_dim=2, embedding_size=32, hidden_size=16, latent_size=8, reward_head=False,
    )
    optimizer = torch.optim.AdamW(plain_model.parameters(), lr=3e-4)
    scheduler = build_lr_scheduler(optimizer, warmup_steps=2, total_steps=10, schedule="cosine")
    sum(p.sum() for p in plain_model.parameters()).backward()
    optimizer.step()
    scheduler.step()
    saved = save_checkpoint(
        checkpoint_dir=config.checkpoint_dir,
        epoch=1,
        model=plain_model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        metrics={"total_loss": 0.0},
        global_step=1,
        history_path=config.checkpoint_dir / "history.jsonl",
    )

    upgraded_model = VisualWorldModel(
        action_dim=2, embedding_size=32, hidden_size=16, latent_size=8, reward_head=True,
    )
    upgraded_optimizer = torch.optim.AdamW(upgraded_model.parameters(), lr=3e-4)
    upgraded_scheduler = build_lr_scheduler(
        upgraded_optimizer, warmup_steps=2, total_steps=10, schedule="cosine",
    )
    # No expected_config so the reward_head config-flag drift is not compared.
    epoch, step = load_checkpoint(
        saved, upgraded_model, upgraded_optimizer, torch.device("cpu"), scheduler=upgraded_scheduler,
    )
    assert epoch == 1 and step == 1
    # Reward head exists and was initialized fresh.
    assert upgraded_model.reward_head is not None
    out = upgraded_model(observations=_observations(), actions=_actions())
    assert out.reward_predictions is not None and out.reward_predictions.shape == (2, 3)


def test_checkpoint_compat_with_to_without_reward_head(tmp_path: Path) -> None:
    """Checkpoint trained with reward head still loads into a plain model."""

    config = TrainConfig(
        dataset_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "ckpt",
        sequence_length=16,
        hidden_size=16,
        latent_size=8,
        embedding_size=32,
        reward_head=True,
    )
    config.checkpoint_dir.mkdir(parents=True)
    full_model = VisualWorldModel(
        action_dim=2, embedding_size=32, hidden_size=16, latent_size=8, reward_head=True,
    )
    optimizer = torch.optim.AdamW(full_model.parameters(), lr=3e-4)
    scheduler = build_lr_scheduler(optimizer, warmup_steps=2, total_steps=10, schedule="cosine")
    sum(p.sum() for p in full_model.parameters()).backward()
    optimizer.step()
    scheduler.step()
    saved = save_checkpoint(
        checkpoint_dir=config.checkpoint_dir,
        epoch=1,
        model=full_model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        metrics={"total_loss": 0.0},
        global_step=1,
        history_path=config.checkpoint_dir / "history.jsonl",
    )

    plain_model = VisualWorldModel(
        action_dim=2, embedding_size=32, hidden_size=16, latent_size=8, reward_head=False,
    )
    plain_optimizer = torch.optim.AdamW(plain_model.parameters(), lr=3e-4)
    plain_scheduler = build_lr_scheduler(
        plain_optimizer, warmup_steps=2, total_steps=10, schedule="cosine",
    )
    epoch, step = load_checkpoint(
        saved, plain_model, plain_optimizer, torch.device("cpu"), scheduler=plain_scheduler,
    )
    assert epoch == 1 and step == 1
    assert plain_model.reward_head is None


def test_constant_scheduler_reaches_peak_after_warmup() -> None:
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    base_lr = 3e-4
    warmup = 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=warmup,
        total_steps=10,
        schedule="constant",
    )
    lrs = [scheduler.get_last_lr()[0]]
    for _ in range(9):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    for prev, cur in zip(lrs[: warmup - 1], lrs[1:warmup]):
        assert cur > prev
    for lr in lrs[warmup - 1 :]:
        assert abs(lr - base_lr) < 1e-7


def test_kl_balancing_detach_semantics() -> None:
    # alpha=1.0 → only KL(sg(q)||p): gradient flows only through the prior.
    torch.manual_seed(0)
    posterior_mean = torch.randn(2, 3, 4, requires_grad=True)
    posterior_std = (torch.rand(2, 3, 4) + 0.2).detach().requires_grad_(True)
    prior_mean = torch.randn(2, 3, 4, requires_grad=True)
    prior_std = (torch.rand(2, 3, 4) + 0.2).detach().requires_grad_(True)

    parts = compute_rssm_kl_loss(
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        prior_mean=prior_mean,
        prior_std=prior_std,
        kl_balance_alpha=1.0,
        kl_free_nats=0.0,
    )
    parts["kl_loss"].backward()

    assert posterior_mean.grad is None or torch.allclose(
        posterior_mean.grad, torch.zeros_like(posterior_mean)
    )
    assert posterior_std.grad is None or torch.allclose(
        posterior_std.grad, torch.zeros_like(posterior_std)
    )
    assert prior_mean.grad is not None and prior_mean.grad.abs().sum() > 0.0
    assert prior_std.grad is not None and prior_std.grad.abs().sum() > 0.0


def test_compute_grad_norm_matches_reference_loop() -> None:
    """``compute_grad_norm`` must match a naive per-parameter sum-of-squares loop.

    Pins the Tier-B refactor: the fused-norm path (single ``.item()`` sync)
    must produce numerically the same scalar as the previous per-parameter
    ``.item()`` loop. Without this test, a future "optimization" could swap
    in a different reduction (e.g. infinity norm) and silently change the
    metric reported in ``history.jsonl``.
    """

    from train.train_rssm import compute_grad_norm

    torch.manual_seed(7)
    module = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    inputs = torch.randn(5, 8)
    targets = torch.randn(5, 4)
    loss = torch.nn.functional.mse_loss(module(inputs), targets)
    loss.backward()

    expected_squared = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        expected_squared += float(p.grad.detach().norm(2).item()) ** 2
    expected = expected_squared ** 0.5

    actual = compute_grad_norm(module)
    assert abs(actual - expected) < 1e-5, f"actual={actual} expected={expected}"


def test_compute_grad_norm_zero_when_no_grads() -> None:
    """Gracefully report zero when every parameter's grad is ``None``."""

    from train.train_rssm import compute_grad_norm

    module = torch.nn.Linear(4, 2)
    assert compute_grad_norm(module) == 0.0
