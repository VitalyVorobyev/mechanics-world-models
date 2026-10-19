"""Tests for the October 2026 gate-experiment options of the mechanics model.

Covers the periodic angle coordinate, the causal / sample-consistent q̇, the
style-only nuisance channel, the passive linear damping, the MLP-dynamics
baseline, and the state probes. Legacy defaults must stay bit-compatible, so
a few tests also pin that the default configuration is unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eval.probes import group_train_test_split, pose_targets, run_state_probes
from models.mechanics import MechanicsWorldModel, compute_mechanics_world_model_loss
from models.mechanics.coordinates import embed_coordinates, wrap_angle, wrap_angle_dims
from models.mechanics.encoder import derive_qdot_from_q
from models.mechanics.lagrangian import LinearDissipation


def _small_model(**kwargs) -> MechanicsWorldModel:
    torch.manual_seed(0)
    defaults = dict(
        action_dim=1,
        q_dim=2,
        nuisance_dim=4,
        embedding_size=32,
        encoder_head_hidden_size=16,
        mass_matrix_hidden_size=16,
        potential_hidden_size=16,
        dissipation_hidden_size=16,
        actuation_hidden_size=16,
        mlp_dynamics_hidden_size=16,
        dynamics_hidden_layers=1,
        image_size=84,
        dt=0.02,
    )
    defaults.update(kwargs)
    return MechanicsWorldModel(**defaults)


GATE_OPTIONS = dict(
    qdot_stencil="backward",
    angle_dims=(1,),
    nuisance_mode="style",
    dissipation_form="linear",
)


def test_wrap_and_embed_coordinates() -> None:
    x = torch.tensor([3.0 * math.pi, -3.5 * math.pi, 0.25])
    wrapped = wrap_angle(x)
    assert torch.all(wrapped >= -math.pi) and torch.all(wrapped < math.pi)
    assert torch.allclose(torch.cos(wrapped), torch.cos(x), atol=1e-6)

    q = torch.tensor([[0.5, 7.0]])
    assert torch.equal(embed_coordinates(q, ()), q)  # legacy: identity
    embedded = embed_coordinates(q, (1,))
    assert torch.allclose(embedded, torch.tensor([[0.5, math.cos(7.0), math.sin(7.0)]]))
    # Only the angle dim wraps.
    out = wrap_angle_dims(torch.tensor([[10.0, 10.0]]), (1,))
    assert out[0, 0] == pytest.approx(10.0)
    assert out[0, 1] == pytest.approx(10.0 - 4 * math.pi)


def test_backward_stencil_is_causal_and_wraps_branch_cut() -> None:
    dt = 0.02
    # Angle crossing +pi -> -pi between frames 1 and 2 is a small step.
    q = torch.tensor([[[0.0, math.pi - 0.02], [0.1, math.pi - 0.01], [0.2, -math.pi + 0.01]]])
    std = torch.full_like(q, 0.05)
    qdot, qdot_std = derive_qdot_from_q(q, std, dt=dt, stencil="backward", angle_dims=(1,))
    assert qdot[0, 2, 1] == pytest.approx(0.02 / dt, rel=1e-4)
    assert qdot[0, 2, 0] == pytest.approx(0.1 / dt, rel=1e-4)
    assert qdot[0, 1, 0] == pytest.approx(0.1 / dt, rel=1e-4)  # uses frames 0, 1 only
    assert torch.allclose(qdot_std, torch.full_like(q, math.sqrt(2) * 0.05 / dt))


def test_default_model_layout_is_legacy() -> None:
    model = _small_model()
    assert model.decoder.feature_size == 2 * 2 + 4
    assert model.feature_size == 2 * 2 + 4
    assert model.style_head is None
    assert model.encoder.mech_head[-1].out_features == 2 * 2
    assert model.dynamics_form == "lagrangian"


def test_imagination_ignores_frames_after_context() -> None:
    """The open-loop start state must not see the first predicted frame."""

    for options in ({}, GATE_OPTIONS):
        model = _small_model(**options).eval()
        torch.manual_seed(1)
        observations = torch.rand(2, 8, 3, 84, 84)
        actions = torch.randn(2, 8, 1)
        perturbed = observations.clone()
        perturbed[:, 4:] = torch.rand_like(perturbed[:, 4:])
        with torch.no_grad():
            first = model.imagine_from_posterior(
                rollout=model._rollout(model.encoder(observations), actions),
                actions=actions, context_steps=4, horizon=4,
            )[0]
            second = model.imagine_from_posterior(
                rollout=model._rollout(model.encoder(perturbed), actions),
                actions=actions, context_steps=4, horizon=4,
            )[0]
        assert torch.allclose(first, second, atol=1e-6), options


def test_style_mode_z_cannot_move_content() -> None:
    model = _small_model(**GATE_OPTIONS)
    assert model.decoder.feature_size == 3  # (x, cos θ, sin θ)
    assert model.feature_size == 3 + 2  # reward features: embedded q + q̇, no z
    assert model.style_head is not None
    with torch.no_grad():
        model.style_head.weight.normal_()
        q = torch.randn(1, 1, 2)
        qdot = torch.randn(1, 1, 2)
        a = model.decode(q, qdot, torch.randn(1, 1, 4))
        b = model.decode(q, qdot, torch.randn(1, 1, 4))
    logit_a = torch.logit(a.clamp(1e-6, 1 - 1e-6))
    logit_b = torch.logit(b.clamp(1e-6, 1 - 1e-6))
    # Per channel, the two renders are an affine map of each other in logit space.
    for channel in range(3):
        x = logit_a[0, 0, channel].flatten().double()
        y = logit_b[0, 0, channel].flatten().double()
        design = torch.stack([x, torch.ones_like(x)], dim=1)
        solution = torch.linalg.lstsq(design, y.unsqueeze(1)).solution
        residual = y - (design @ solution).squeeze(1)
        assert residual.abs().max() < 1e-3


def test_linear_dissipation_is_passive() -> None:
    torch.manual_seed(0)
    damping = LinearDissipation(q_dim=3)
    with torch.no_grad():
        damping.raw.normal_()
    qdot = torch.randn(256, 3) * 10.0
    power = (qdot * damping.force(qdot)).sum(dim=-1)
    assert torch.all(power >= -1e-6)


@pytest.mark.parametrize(
    "options",
    [GATE_OPTIONS, {**GATE_OPTIONS, "dynamics_form": "mlp"}],
    ids=["lagrangian", "mlp"],
)
def test_gate_options_train_step_is_finite(options: dict) -> None:
    model = _small_model(**options, reward_head=True)
    observations = torch.rand(2, 8, 3, 84, 84)
    actions = torch.randn(2, 8, 1)
    rewards = torch.rand(2, 8)
    outputs = model(
        observations=observations, actions=actions,
        imagination_context_steps=3, imagination_horizon=4,
    )
    losses = compute_mechanics_world_model_loss(
        outputs, observations, next_observations=torch.rand(2, 8, 3, 84, 84),
        rewards=rewards, foreground_imagination_reconstruction_weight=1.0,
        reward_weight=1.0, imagined_reward_weight=1.0,
    )
    losses["total_loss"].backward()
    assert torch.isfinite(losses["total_loss"])
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # Angle posterior lives on the circle; prior means are re-centred near it.
    q_mean = outputs.rollout.posterior.q_mean
    assert torch.all(q_mean[..., 1].abs() <= math.pi + 1e-5)
    gap = (outputs.rollout.prior_q_mean - q_mean)[..., 1]
    assert torch.all(gap.abs() <= math.pi + 1e-5)


def test_qdot_samples_follow_q_samples() -> None:
    model = _small_model(qdot_stencil="backward")
    observations = torch.rand(2, 6, 3, 84, 84)
    rollout = model._rollout(model.encoder(observations), torch.randn(2, 6, 1))
    expected, _ = derive_qdot_from_q(
        rollout.q_sampled, rollout.posterior.q_std, dt=model.dt, stencil="backward",
    )
    assert torch.allclose(rollout.qdot_sampled, expected)


def test_pose_targets_and_group_split() -> None:
    qpos = np.array([[0.1, 0.0], [0.2, math.pi / 2]])
    targets, names = pose_targets(qpos, (1,))
    assert names == ["qpos0", "cos_qpos1", "sin_qpos1"]
    np.testing.assert_allclose(targets, [[0.1, 1.0, 0.0], [0.2, 0.0, 1.0]], atol=1e-12)

    groups = np.repeat(np.arange(5), 10)
    train_idx, test_idx, kind = group_train_test_split(groups, seed=0)
    assert kind == "episode"
    assert not set(groups[train_idx]) & set(groups[test_idx])
    _, _, kind = group_train_test_split(np.zeros(10, dtype=np.int64))
    assert kind == "row"


def test_state_probes_separate_informative_and_uninformative_latents() -> None:
    rng = np.random.default_rng(0)
    batch, steps = 12, 20
    x = rng.uniform(-1.5, 1.5, size=(batch, steps, 1))
    theta = rng.uniform(-30.0, 30.0, size=(batch, steps, 1))  # unbounded raw angle
    qpos = np.concatenate([x, theta], axis=-1)
    qvel = rng.standard_normal(size=(batch, steps, 2))
    informative = np.concatenate([2.0 * x + 1.0, np.cos(theta), np.sin(theta)], axis=-1)
    result = run_state_probes(
        {"good": informative, "noise": rng.standard_normal(size=(batch, steps, 3))},
        qpos, qvel, list(range(batch)),
    )
    assert result["split"] == "episode"
    assert result["probes"]["good"]["pose"]["linear"]["r2_overall"] > 0.99
    assert result["probes"]["good"]["pose"]["mlp"]["r2_overall"] > 0.9
    assert result["probes"]["noise"]["pose"]["linear"]["r2_overall"] < 0.1
