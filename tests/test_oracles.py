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
from oracles.dynamics import _windows, free_segments, run_dynamics_oracle


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


def test_dynamics_oracle_runs_end_to_end(tmp_path: Path) -> None:
    kwargs = dict(
        ks=(1,),
        segment_length=20,
        train_horizon=4,
        pretrain_steps=5,
        adapt_lrs=(1e-3,),
        adapt_checkpoints=(0, 2),
        batch_size=8,
        eval_horizons=(5, 20),
        rail_margin=None,
        cache_dir=tmp_path,
        log=lambda _: None,
    )
    shifted = {
        "heavy": tuple(_synthetic_episodes(n, 60, 2.0, s) for n, s in ((3, 2), (2, 3), (2, 4))),
    }
    result = run_dynamics_oracle(
        _synthetic_episodes(4, 60, 1.0, 0), _synthetic_episodes(2, 60, 1.0, 1), shifted, **kwargs,
    )
    for mode in ("lag_low", "lag_full", "mlp_last", "mlp_full"):
        assert np.isfinite(result.metrics[f"heavy.k1.{mode}.dev_h20"])
        assert np.isfinite(result.metrics[f"heavy.k1.{mode}.cellbest.dev_h20"])
    assert "mlp_over_lag_error_ratio_k1_h20" in result.metrics
    assert "prereq_nominal_over_shift_effect" in result.metrics
    # Re-running with the same config resumes from the cache and reproduces the metrics.
    again = run_dynamics_oracle(
        _synthetic_episodes(4, 60, 1.0, 0), _synthetic_episodes(2, 60, 1.0, 1), shifted, **kwargs,
    )
    assert again.metrics == pytest.approx(result.metrics)
    assert pose_features(np.zeros((1, 2))).shape == (1, 3)


def test_rail_windows_are_excluded() -> None:
    episodes = _synthetic_episodes(2, 40, 1.0, 0)
    episodes.qpos[0, 30, 0] = 1.75  # cart reaches the rail in episode 0
    all_windows = _windows(episodes, 5)
    free = _windows(episodes, 5, rail_margin=0.2)
    assert free[0].shape[0] == all_windows[0].shape[0] - 6
    assert free_segments(episodes, 10, 0.2, seed=0) and (0, 20) not in free_segments(episodes, 10, 0.2, seed=0)


def test_true_dynamics_ceiling_is_small_away_from_rail() -> None:
    pytest.importorskip("dm_control")
    from envs.dm_control_pixels import DmControlPixelEnv, PixelEnvConfig
    from oracles.ceiling import run_dynamics_ceiling

    env = DmControlPixelEnv(PixelEnvConfig(seed=0))
    env.reset()
    rng = np.random.default_rng(0)
    states, actions = [env.physics_state()], []
    for _ in range(80):
        actions.append(rng.uniform(-1, 1, size=(1,)))
        states.append(env.step_without_render(actions[-1]))
    env.close()
    episodes = StateEpisodes(
        qpos=np.stack([s["qpos"] for s in states])[None],
        qvel=np.stack([s["qvel"] for s in states])[None],
        actions=np.stack(actions)[None],
        frame_dt=env.frame_dt(),
    )
    result = run_dynamics_ceiling({"nominal": (episodes, None)}, substeps=(4,), horizons=(10, 20), log=lambda _: None)
    assert result.metrics["nominal.sub4.free.dev_h20"] < 0.01


def test_intervention_effect_detects_cart_mass_not_pole_mass() -> None:
    pytest.importorskip("dm_control")
    from oracles.intervention import run_intervention_effect

    result = run_intervention_effect(
        {"pole_mass_x2": {"body_mass_2": 2.0}, "cart_mass_x05": {"body_mass_1": 0.5}},
        seeds=range(2),
        horizons=(10, 20),
    )
    assert result.metrics["cart_mass_x05.mean_abs_dev_h20"] > 5 * result.metrics["pole_mass_x2.mean_abs_dev_h20"]


def test_actuated_dims_zero_unactuated_rows() -> None:
    from models.mechanics.lagrangian import Actuation

    actuation = Actuation(q_dim=2, action_dim=1, hidden_size=8, angle_dims=(1,), actuated_dims=(0,))
    with torch.no_grad():
        b = actuation(torch.randn(5, 2))
    assert b.shape == (5, 2, 1)
    assert torch.all(b[:, 1] == 0) and torch.any(b[:, 0] != 0)
    with pytest.raises(ValueError):
        Actuation(q_dim=2, action_dim=1, actuated_dims=())


def test_context_oracle_runs_end_to_end(tmp_path: Path) -> None:
    from oracles.context import run_context_oracle

    train_envs = {"nominal": _synthetic_episodes(3, 60, 1.0, 0), "light": _synthetic_episodes(3, 60, 0.7, 5)}
    shifted = {"heavy": tuple(_synthetic_episodes(n, 60, 2.0, s) for n, s in ((3, 2), (2, 3), (2, 4)))}
    result = run_context_oracle(
        train_envs, _synthetic_episodes(2, 60, 1.0, 1), shifted, ks=(1,), segment_length=20, train_horizon=4,
        pretrain_steps=3, adapt_lrs=(1e-3,), adapt_checkpoints=(0, 2), batch_size=8, eval_horizons=(5, 20),
        rail_margin=None, cache_dir=tmp_path, log=lambda _: None,
    )
    for mode in ("lag_ctx", "lag_full", "mlp_ctx", "mlp_full"):
        assert np.isfinite(result.metrics[f"heavy.k1.{mode}.dev_h20"])
    assert "mlp_ctx_over_lag_ctx_ratio_k1_h20" in result.metrics


def test_noisy_observation_uses_backward_differences() -> None:
    from oracles.common import noisy_observation

    clean = _synthetic_episodes(2, 30, 1.0, 0)
    noisy = noisy_observation(clean, 0.01, 0.02, seed=0)
    assert np.abs(noisy.qpos - clean.qpos).std() > 0
    expected = (noisy.qpos[:, 5] - noisy.qpos[:, 4]) / clean.frame_dt
    assert np.allclose(noisy.qvel[:, 5], expected)
    assert np.array_equal(noisy.actions, clean.actions)


def test_context_oracle_with_pose_noise_reports_noisy_start(tmp_path: Path) -> None:
    from oracles.context import run_context_oracle

    train_envs = {"nominal": _synthetic_episodes(3, 60, 1.0, 0), "light": _synthetic_episodes(3, 60, 0.7, 5)}
    shifted = {"heavy": tuple(_synthetic_episodes(n, 60, 2.0, s) for n, s in ((3, 2), (2, 3), (2, 4)))}
    result = run_context_oracle(
        train_envs, _synthetic_episodes(2, 60, 1.0, 1), shifted, ks=(1,), segment_length=20, train_horizon=4,
        pretrain_steps=3, adapt_lrs=(1e-3,), adapt_checkpoints=(0, 2), batch_size=8, eval_horizons=(5, 20),
        rail_margin=None, modes=("lag_ctx", "mlp_ctx"), pose_noise=(0.01, 0.02), cache_dir=tmp_path,
        log=lambda _: None,
    )
    assert np.isfinite(result.metrics["heavy.k1.lag_ctx.noisy_start.dev_h20"])
    assert "mlp_ctx_over_lag_ctx_ratio.noisy_start_k1_h20" in result.metrics
    assert result.config["pose_noise"] == [0.01, 0.02]


def test_simulated_state_episodes_have_collector_layout() -> None:
    pytest.importorskip("dm_control")
    from oracles.context import simulate_state_episodes

    episodes = simulate_state_episodes({"body_mass_1": 0.7}, episodes=1, seed=0)
    assert episodes.qpos.shape == (1, 501, 2) and episodes.actions.shape == (1, 500, 1)
    assert episodes.frame_dt == pytest.approx(0.02)


def test_true_mass_matrix_matches_cartpole_parameters() -> None:
    pytest.importorskip("dm_control")
    from oracles.gauge import true_mass_matrices

    m = true_mass_matrices(np.array([[0.0, 0.0], [0.5, math.pi / 2]]))
    assert m[:, 0, 0] == pytest.approx([1.1, 1.1], rel=1e-3)  # cart 1.0 + pole 0.1
    assert abs(m[1, 0, 1]) < 1e-6 < abs(m[0, 0, 1])  # coupling ~ cos(theta)


def test_pose_errors_wrap_angles_and_convert_to_pixels() -> None:
    from oracles.encoder import PIXELS_PER_METRE, pose_errors

    qpos = np.array([[0.0, math.pi - 0.01], [1.0, 0.0]])
    theta = np.array([-math.pi + 0.01, 0.02])  # wraps: errors 0.02 rad each
    pred = np.stack([qpos[:, 0] + 0.1, np.cos(theta), np.sin(theta)], axis=-1)
    errors = pose_errors(pred, qpos)
    assert errors["theta_err_deg_median"] == pytest.approx(math.degrees(0.02), rel=1e-6)
    assert errors["x_err_px_rms"] == pytest.approx(0.1 * PIXELS_PER_METRE)
    assert errors["tip_err_px_rms"] == pytest.approx(0.02 * PIXELS_PER_METRE, rel=1e-6)


def _particle_rollout(q0: np.ndarray, v0: np.ndarray, actions: np.ndarray, dt: float = 0.02) -> np.ndarray:
    q, v, out = q0.copy(), v0.copy(), []
    for step in range(actions.shape[1]):
        v = v + dt * (actions[:, step, :1] - 0.5 * q)
        q = q + dt * v
        out.append(q)
    return np.stack(out, axis=1)


def test_model_based_start_state_beats_backward_difference() -> None:
    from oracles.common import StateEpisodes, noisy_observation
    from oracles.start_state import context_windows, evaluate_starts

    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(4, 80, 1))
    q0, v0 = rng.normal(size=(4, 2)), rng.normal(size=(4, 2))
    qpos = np.concatenate([q0[:, None], _particle_rollout(q0, v0, actions)], axis=1)
    qvel = np.gradient(qpos, 0.02, axis=1)
    clean = StateEpisodes(qpos=qpos, qvel=qvel, actions=actions, frame_dt=0.02)
    noisy = noisy_observation(clean, 0.01, 0.01, seed=1)
    windows = context_windows(clean, noisy, horizon=20, context=8, stride=5, rail_margin=None)
    errors = evaluate_starts(_particle_rollout, windows, (20,), sigma=(0.01, 0.01))
    assert errors["shoot.dev_h20"] < 0.5 * errors["backward.dev_h20"]
    assert errors["quadfit.dev_h20"] < errors["backward.dev_h20"]


def test_context_oracle_start_state_evaluation(tmp_path: Path) -> None:
    from oracles.context import run_context_oracle

    train_envs = {"nominal": _synthetic_episodes(3, 60, 1.0, 0), "light": _synthetic_episodes(3, 60, 0.7, 5)}
    shifted = {"heavy": tuple(_synthetic_episodes(n, 60, 2.0, s) for n, s in ((3, 2), (2, 3), (2, 4)))}
    kwargs = dict(
        ks=(1,), segment_length=20, train_horizon=4, pretrain_steps=3, adapt_lrs=(1e-3,), adapt_checkpoints=(0, 2),
        batch_size=8, eval_horizons=(5, 20), rail_margin=None, modes=("lag_ctx", "mlp_ctx"),
        pose_noise=(0.01, 0.02), cache_dir=tmp_path, log=lambda _: None,
    )
    plain = run_context_oracle(train_envs, _synthetic_episodes(2, 60, 1.0, 1), shifted, **kwargs)
    result = run_context_oracle(train_envs, _synthetic_episodes(2, 60, 1.0, 1), shifted, start_context=4, **kwargs)
    assert np.isfinite(result.metrics["heavy.k1.lag_ctx.start.shoot.dev_h20"])
    assert "start_ratio.shoot_k1_h20" in result.metrics
    # Evaluation-only option: the adaptation cells are reused, so headline metrics are identical.
    assert result.metrics["heavy.k1.lag_ctx.dev_h20"] == plain.metrics["heavy.k1.lag_ctx.dev_h20"]


def test_savgol_velocity_is_less_noisy_than_central() -> None:
    from oracles.common import noisy_observation

    clean = _synthetic_episodes(2, 120, 1.0, 0)
    errors = {}
    for stencil in ("central", "savgol"):
        observed = noisy_observation(clean, 0.01, 0.01, seed=0, velocity=stencil)
        reference = noisy_observation(clean, 0.0, 0.0, seed=0, velocity=stencil)  # same stencil, no noise
        errors[stencil] = (observed.qvel - reference.qvel)[:, 10:-10].std()
    assert errors["savgol"] < 0.3 * errors["central"]


def test_context_oracle_uses_observed_poses_and_valid_frames(tmp_path: Path) -> None:
    from oracles.context import run_context_oracle

    def observed(episodes):
        rng = np.random.default_rng(0)
        valid = np.ones(episodes.qpos.shape[:2], dtype=bool)
        valid[:, 30:35] = False  # e.g. out of view
        episodes.observed_qpos = episodes.qpos + 0.01 * rng.normal(size=episodes.qpos.shape)
        episodes.valid = valid
        return episodes

    train_envs = {"nominal": observed(_synthetic_episodes(3, 60, 1.0, 0)), "light": observed(_synthetic_episodes(3, 60, 0.7, 5))}
    shifted = {"heavy": tuple(observed(_synthetic_episodes(n, 60, 2.0, s)) for n, s in ((3, 2), (2, 3), (2, 4)))}
    result = run_context_oracle(
        train_envs, _synthetic_episodes(2, 60, 1.0, 1), shifted, ks=(1,), segment_length=20, train_horizon=4,
        pretrain_steps=3, adapt_lrs=(1e-3,), adapt_checkpoints=(0, 2), batch_size=8, eval_horizons=(5, 20),
        rail_margin=None, modes=("lag_ctx", "mlp_ctx"), pose_source="test", start_context=4, cache_dir=tmp_path,
        log=lambda _: None,
    )
    assert result.config["pose_source"] == "test"
    assert np.isfinite(result.metrics["heavy.k1.mlp_ctx.start.shoot.dev_h20"])


def test_windows_drop_invalid_frames() -> None:
    from oracles.dynamics import _windows

    episodes = _synthetic_episodes(1, 40, 1.0, 0)
    full = _windows(episodes, 5)[0].shape[0]
    episodes.valid = np.ones((1, 41), dtype=bool)
    episodes.valid[0, 20] = False
    assert _windows(episodes, 5)[0].shape[0] == full - 6  # starts 15..20 contain frame 20


def test_rendered_states_feed_the_pose_encoder() -> None:
    pytest.importorskip("dm_control")
    from oracles.encoder import _PoseEncoder
    from oracles.pixel_pose import with_encoder_poses

    episodes = _synthetic_episodes(1, 6, 1.0, 0)
    out = with_encoder_poses(episodes, None, _PoseEncoder("conv").eval())
    assert out.observed_qpos.shape == episodes.qpos.shape and out.valid.shape == episodes.qpos.shape[:2]


def test_frozen_renderer_oracle_runs() -> None:
    from oracles.frozen_render import run_frozen_renderer_oracle

    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(8, 84, 84, 3), dtype=np.uint8)
    qpos = rng.normal(size=(8, 2))
    result = run_frozen_renderer_oracle(
        images, qpos, images, qpos, decoder_steps=2, encoder_steps=2, batch_size=4, log=lambda _: None,
    )
    assert np.isfinite(result.metrics["theta_err_deg_median"])


def test_pooled_loss_gives_gradient_without_overlap() -> None:
    import torch

    from oracles.frozen_render import _reconstruction_loss

    target = torch.zeros(1, 3, 84, 84)
    target[..., 60:80, 40:44] = 1.0  # pole hanging
    shift = torch.tensor(0.0, requires_grad=True)
    rows, cols = torch.arange(84.0)[:, None], torch.arange(84.0)[None, :]
    column = ((cols >= 40) & (cols < 44)).float()
    pred = (torch.exp(-((rows - 20.0 - 40.0 * shift) ** 2) / 20.0) * column)[None, None].expand(1, 3, 84, 84)  # pole up
    plain = torch.autograd.grad(_reconstruction_loss(pred, target, 1.0), shift, retain_graph=True)[0]
    pooled = torch.autograd.grad(_reconstruction_loss(pred, target, 1.0, (24,)), shift)[0]
    assert abs(float(pooled)) > 10 * abs(float(plain))


def test_context_oracle_potential_slot(tmp_path: Path) -> None:
    from oracles.context import run_context_oracle

    train_envs = {"nominal": _synthetic_episodes(3, 60, 1.0, 0), "light": _synthetic_episodes(3, 60, 0.7, 5)}
    shifted = {"heavy": tuple(_synthetic_episodes(n, 60, 2.0, s) for n, s in ((3, 2), (2, 3), (2, 4)))}
    result = run_context_oracle(
        train_envs, _synthetic_episodes(2, 60, 1.0, 1), shifted, ks=(1,), segment_length=20, train_horizon=4,
        pretrain_steps=3, adapt_lrs=(1e-3,), adapt_checkpoints=(0, 2), batch_size=8, eval_horizons=(5, 20),
        rail_margin=None, modes=("lag_ctx", "mlp_ctx"), potential_slot=True, cache_dir=tmp_path, log=lambda _: None,
    )
    assert result.config["potential_slot"] is True
    assert np.isfinite(result.metrics["heavy.k1.lag_ctx.dev_h20"])


def test_pose_calibration_removes_pose_dependent_bias() -> None:
    from oracles.pixel_pose import PoseCalibration

    episodes = _synthetic_episodes(4, 400, 1.0, 0)
    bias = np.stack([0.02 * np.cos(episodes.qpos[..., 1]), 0.05 * np.sin(episodes.qpos[..., 1])], axis=-1)
    episodes.observed_qpos = episodes.qpos + bias
    corrected = PoseCalibration(theta_bins=48).fit(episodes).apply(episodes)
    assert np.abs(corrected.observed_qpos - episodes.qpos).mean() < 0.3 * np.abs(bias).mean()


def test_heuristic_poses_read_rendered_frames() -> None:
    pytest.importorskip("dm_control")
    from oracles.bootstrap import heuristic_poses
    from oracles.pixel_pose import render_states

    qpos = np.array([[0.3, math.pi], [-0.5, 0.4], [0.0, -1.2]])
    poses, found = heuristic_poses(render_states(qpos, None))
    assert found.all()
    assert np.abs(poses[:, 0] - qpos[:, 0]).max() < 0.06
    assert np.abs(np.angle(np.exp(1j * (poses[:, 1] - qpos[:, 1])))).max() < math.radians(5)


def test_reject_jumps_flags_spikes() -> None:
    from oracles.common import reject_jumps

    episodes = _synthetic_episodes(1, 200, 1.0, 0)
    rng = np.random.default_rng(0)
    episodes.observed_qpos = episodes.qpos + 0.002 * rng.normal(size=episodes.qpos.shape)
    episodes.observed_qpos[0, 100, 1] += 0.3
    valid = reject_jumps(episodes).valid
    assert not valid[0, 100] and valid[0, 50] and valid.mean() > 0.9


def test_reject_jumps_scale_floor_handles_quantized_poses() -> None:
    from oracles.common import reject_jumps

    episodes = _synthetic_episodes(1, 200, 1.0, 0)
    episodes.observed_qpos = episodes.qpos.copy()
    episodes.observed_qpos[0, :, 0] = 0.01 * (np.arange(201) // 7)  # cart moving in pixel-sized steps
    assert reject_jumps(episodes, 6.0).valid.mean() < 0.5  # MAD collapses to zero: every step is a "jump"
    assert reject_jumps(episodes, 6.0, (0.004, 0.012)).valid.mean() > 0.95
