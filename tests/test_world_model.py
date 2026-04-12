from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models import (
    ConvEncoder,
    ImageDecoder,
    RSSM,
    VisualWorldModel,
    compute_latent_consistency_loss,
    compute_world_model_loss,
    foreground_reconstruction_mask,
)
from train.train_rssm import build_arg_parser, serializable_config
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
        "dynamic_reconstruction_loss",
        "weighted_dynamic_reconstruction_loss",
        "kl_loss",
        "weighted_kl_loss",
        "latent_consistency_loss",
        "weighted_latent_consistency_loss",
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
    assert serialized["reconstruction_weight"] == 0.05
    assert serialized["foreground_reconstruction_weight"] == 2.0
    assert serialized["foreground_mask_floor"] == 0.02
    assert serialized["foreground_mask_kernel_size"] == 7
    assert serialized["transition_reconstruction_weight"] == 1.5
    assert serialized["foreground_transition_reconstruction_weight"] == 2.0
    assert serialized["dynamic_reconstruction_weight"] == 1.0
    assert serialized["dynamic_mask_floor"] == 0.1
    assert serialized["latent_consistency_weight"] == 1.0
