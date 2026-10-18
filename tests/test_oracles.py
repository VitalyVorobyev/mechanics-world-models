"""Tests for the oracle harness and the keypoint / renderer front end."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from models.mechanics import MechanicsWorldModel, compute_mechanics_world_model_loss
from models.mechanics.keypoints import GeometricRenderer, keypoints_to_cartpole_q
from oracles.common import Criterion, OracleResult, StateEpisodes, pose_features
from oracles.dynamics import run_dynamics_oracle


def test_oracle_result_verdict_and_write(tmp_path: Path) -> None:
    result = OracleResult(
        name="demo",
        hypotheses=["H0"],
        question="?",
        config={"a": 1},
        metrics={"err": 0.1, "ratio": np.float64(2.0)},
        criteria=[Criterion("err", "<=", 0.2), Criterion("ratio", ">=", 1.5)],
    )
    assert result.verdict()["passed"] is True
    path = result.write(root=tmp_path)
    payload = json.loads(path.read_text())
    assert payload["verdict"]["passed"] is True
    assert (tmp_path / "demo" / "latest.json").exists()
    missing = OracleResult(name="m", hypotheses=[], question="?", config={}, criteria=[Criterion("x", "<", 1)])
    assert missing.verdict()["passed"] is None


def test_renderer_keypoints_invert_through_geometry() -> None:
    renderer = GeometricRenderer()
    q = torch.tensor([[0.3, 0.0], [-0.2, 0.7], [0.1, -2.5]])
    with torch.no_grad():
        keypoints = renderer.keypoints(q)
    # theta = 0 puts the tip straight above the cart (smaller v).
    assert keypoints[0, 1, 0] == pytest.approx(0.3)
    assert keypoints[0, 1, 1] < keypoints[0, 0, 1]
    recovered = keypoints_to_cartpole_q(keypoints)
    assert torch.allclose(recovered[:, 0], q[:, 0], atol=1e-6)
    assert torch.allclose(torch.cos(recovered[:, 1]), torch.cos(q[:, 1]), atol=1e-5)
    assert torch.allclose(torch.sin(recovered[:, 1]), torch.sin(q[:, 1]), atol=1e-5)
    with torch.no_grad():
        assert renderer(q).shape == (3, 3, 84, 84)


def test_keypoint_render_model_losses_backprop() -> None:
    torch.manual_seed(0)
    model = MechanicsWorldModel(
        action_dim=1, q_dim=2, nuisance_dim=4, dt=0.02, angle_dims=(1,), nuisance_mode="style",
        qdot_stencil="backward", dissipation_form="linear", encoder_type="keypoint",
        decoder_type="render", mass_matrix_hidden_size=16, potential_hidden_size=16,
        actuation_hidden_size=16, dynamics_hidden_layers=1,
    )
    observations = torch.rand(2, 10, 3, 84, 84)
    actions = torch.randn(2, 10, 1)
    outputs = model(
        observations, actions, imagination_context_steps=3, imagination_horizon=4,
        latent_dynamics_horizon=4,
    )
    assert outputs.rollout.posterior.keypoints.shape == (2, 10, 2, 2)
    losses = compute_mechanics_world_model_loss(
        outputs, observations, torch.rand(2, 10, 3, 84, 84),
        foreground_imagination_reconstruction_weight=1.0, kl_mech_scale=0.0,
        latent_dynamics_weight=1.0, keypoint_consistency_weight=1.0,
    )
    losses["total_loss"].backward()
    assert torch.isfinite(losses["total_loss"])
    assert float(losses["latent_dynamics_loss"].detach()) >= 0.0
    with pytest.raises(ValueError, match="did not compute"):
        compute_mechanics_world_model_loss(
            model(observations, actions), observations, latent_dynamics_weight=1.0,
        )


def test_render_decoder_requires_cartpole_layout() -> None:
    with pytest.raises(ValueError, match="cartpole-specific"):
        MechanicsWorldModel(action_dim=1, q_dim=3, nuisance_mode="style", decoder_type="render")


def _synthetic_episodes(num: int, frames: int, mass_scale: float, seed: int) -> StateEpisodes:
    """Damped-pendulum-on-a-cart toy trajectories (only shapes/plumbing matter)."""

    rng = np.random.default_rng(seed)
    qpos = np.zeros((num, frames + 1, 2))
    qvel = np.zeros((num, frames + 1, 2))
    actions = rng.uniform(-1, 1, size=(num, frames, 1))
    qpos[:, 0, 1] = rng.uniform(-math.pi, math.pi, size=num)
    dt = 0.02
    for t in range(frames):
        acc_x = actions[:, t, 0] / mass_scale - 0.1 * qvel[:, t, 0]
        acc_th = -9.8 * np.sin(qpos[:, t, 1]) - 0.1 * qvel[:, t, 1]
        qvel[:, t + 1] = qvel[:, t] + dt * np.stack([acc_x, acc_th], axis=-1)
        qpos[:, t + 1] = qpos[:, t] + dt * qvel[:, t + 1]
    return StateEpisodes(qpos=qpos, qvel=qvel, actions=actions, frame_dt=dt)


def test_dynamics_oracle_runs_end_to_end() -> None:
    result = run_dynamics_oracle(
        _synthetic_episodes(4, 60, 1.0, 0),
        _synthetic_episodes(2, 60, 1.0, 1),
        {"heavy": (_synthetic_episodes(3, 60, 2.0, 2), _synthetic_episodes(2, 60, 2.0, 3))},
        ks=(1,),
        segment_length=20,
        train_horizon=4,
        pretrain_steps=5,
        adapt_steps=2,
        batch_size=8,
        eval_horizons=(5, 20),
        log=lambda _: None,
    )
    for mode in ("lag_low", "lag_full", "mlp_last", "mlp_full"):
        assert np.isfinite(result.metrics[f"heavy.k1.{mode}.dev_h20"])
    assert "mlp_over_lag_error_ratio_k1_h20" in result.metrics
    assert pose_features(np.zeros((1, 2))).shape == (1, 3)


def test_intervention_effect_detects_cart_mass_not_pole_mass() -> None:
    pytest.importorskip("dm_control")
    from oracles.intervention import run_intervention_effect

    result = run_intervention_effect(
        {"pole_mass_x2": {"body_mass_2": 2.0}, "cart_mass_x05": {"body_mass_1": 0.5}},
        seeds=range(2),
        horizons=(10, 20),
    )
    assert result.metrics["cart_mass_x05.mean_abs_dev_h20"] > 5 * result.metrics["pole_mass_x2.mean_abs_dev_h20"]
