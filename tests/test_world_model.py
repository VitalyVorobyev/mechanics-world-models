from __future__ import annotations

import torch

from models import ConvEncoder, ImageDecoder, RSSM, VisualWorldModel, compute_world_model_loss


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


def test_world_model_loss_returns_finite_values() -> None:
    observations = _observations()
    actions = _actions()
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)

    outputs = model(observations=observations, actions=actions)
    losses = compute_world_model_loss(outputs, observations, kl_weight=0.1)

    assert set(losses) == {"total_loss", "reconstruction_loss", "kl_loss", "weighted_kl_loss"}
    assert torch.allclose(losses["weighted_kl_loss"], 0.1 * losses["kl_loss"])
    for loss in losses.values():
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_short_training_step_runs_without_nans() -> None:
    torch.manual_seed(0)
    observations = _observations()
    actions = _actions()
    model = VisualWorldModel(action_dim=2, embedding_size=32, hidden_size=16, latent_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    outputs = model(observations=observations, actions=actions)
    losses = compute_world_model_loss(outputs, observations, kl_weight=0.1)
    optimizer.zero_grad(set_to_none=True)
    losses["total_loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    assert torch.isfinite(losses["total_loss"])
    for parameter in model.parameters():
        assert torch.all(torch.isfinite(parameter))
