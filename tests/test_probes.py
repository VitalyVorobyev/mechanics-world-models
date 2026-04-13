"""Linear probe + learned energy diagnostic tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eval.probes import (
    compute_learned_energy,
    encode_dataset,
    fit_linear_probe,
    flatten_target,
    normalized_energy_drift,
)
from models.mechanics import MechanicsWorldModel


def test_linear_probe_perfect_recovery_on_linear_data() -> None:
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=(n, 4))
    w = rng.normal(size=(4, 2))
    b = rng.normal(size=(2,))
    y = x @ w + b
    result = fit_linear_probe(x, y, train_fraction=0.8, seed=0)
    assert result.r2_overall > 0.999
    assert (result.r2_per_dim > 0.999).all()


def test_linear_probe_low_r2_on_random_data() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 3))
    y = rng.normal(size=(300, 2))  # uncorrelated
    result = fit_linear_probe(x, y, train_fraction=0.7, seed=42)
    # Should be close to zero (and may be slightly negative on test).
    assert result.r2_overall < 0.2


def test_linear_probe_validates_shapes() -> None:
    with pytest.raises(ValueError):
        fit_linear_probe(np.zeros((5,)), np.zeros((5, 1)))
    with pytest.raises(ValueError):
        fit_linear_probe(np.zeros((5, 2)), np.zeros((4, 1)))
    with pytest.raises(ValueError):
        fit_linear_probe(np.zeros((10, 2)), np.zeros((10, 1)), train_fraction=1.5)


def test_encode_dataset_returns_flattened_arrays() -> None:
    torch.manual_seed(0)
    model = MechanicsWorldModel(
        action_dim=1,
        q_dim=2,
        nuisance_dim=4,
        embedding_size=32,
        encoder_head_hidden_size=16,
        mass_matrix_hidden_size=16,
        potential_hidden_size=16,
        dissipation_hidden_size=16,
        actuation_hidden_size=16,
        dynamics_hidden_layers=1,
        image_size=84,
        dt=0.02,
    )
    obs = torch.rand(3, 5, 3, 84, 84)
    q, qdot, z = encode_dataset(model, obs, device=torch.device("cpu"))
    assert q.shape == (15, 2)
    assert qdot.shape == (15, 2)
    assert z.shape == (15, 4)


def test_compute_learned_energy_shape_and_finiteness() -> None:
    torch.manual_seed(0)
    model = MechanicsWorldModel(
        action_dim=1,
        q_dim=2,
        nuisance_dim=4,
        embedding_size=32,
        encoder_head_hidden_size=16,
        mass_matrix_hidden_size=16,
        potential_hidden_size=16,
        dissipation_hidden_size=16,
        actuation_hidden_size=16,
        dynamics_hidden_layers=1,
        image_size=84,
        dt=0.02,
    )
    q = torch.randn(2, 4, 2)
    qdot = torch.randn(2, 4, 2)
    e = compute_learned_energy(model, q, qdot)
    assert e.shape == (2, 4)
    assert np.isfinite(e).all()


def test_normalized_energy_drift_zero_at_t0() -> None:
    energy = np.array([[1.0, 1.1, 1.05], [-2.0, -2.0, -2.5]])
    drift = normalized_energy_drift(energy)
    assert drift.shape == energy.shape
    assert np.allclose(drift[:, 0], 0.0)
    assert drift[1, 2] == pytest.approx(-0.25)


def test_flatten_target_validates_shape() -> None:
    with pytest.raises(ValueError):
        flatten_target(np.zeros((4, 2)))
    flat = flatten_target(np.zeros((2, 3, 4)))
    assert flat.shape == (6, 4)
