"""End-to-end smoke test for ``eval-mechanics``.

Builds a tiny synthetic dataset (with optional ``qpos``/``qvel`` so the
linear-probe code path is exercised), trains a ``MechanicsWorldModel`` for
zero steps, saves a checkpoint, and runs the evaluator. The test asserts
that:

* ``metrics.json`` exists and contains the expected MSE keys plus the
  optional probe / energy sub-dicts;
* per-sequence MP4 + PNG visualizations are written;
* the linear-probe section is populated when physics fields are present.

There are no thresholds on metric values — random initialization gives
random predictions; we only check the pipeline runs end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.trajectory import Trajectory, save_trajectory_npz
from eval.eval_mechanics import (
    MechanicsEvalConfig,
    evaluate_checkpoint,
)
from models.mechanics import MechanicsWorldModel


def _make_episode(
    seed: int,
    *,
    image_size: int = 84,
    num_steps: int = 32,
    with_physics: bool = True,
) -> Trajectory:
    rng = np.random.default_rng(seed)
    images = (rng.integers(0, 255, size=(num_steps + 1, image_size, image_size, 3))
              .astype(np.uint8))
    actions = rng.standard_normal(size=(num_steps, 1)).astype(np.float32)
    rewards = rng.standard_normal(size=(num_steps,)).astype(np.float32)
    discounts = np.ones(num_steps, dtype=np.float32)
    dones = np.zeros(num_steps, dtype=np.bool_)
    dones[-1] = True
    step_indices = np.arange(num_steps, dtype=np.int64)
    qpos = qvel = None
    if with_physics:
        qpos = rng.standard_normal(size=(num_steps + 1, 2)).astype(np.float64)
        qvel = rng.standard_normal(size=(num_steps + 1, 2)).astype(np.float64)
    return Trajectory(
        env_name="cartpole-swingup",
        seed=seed,
        episode_id=seed,
        action_repeat=1,
        image_size=image_size,
        images=images,
        actions=actions,
        rewards=rewards,
        discounts=discounts,
        dones=dones,
        step_indices=step_indices,
        qpos=qpos,
        qvel=qvel,
    )


def _save_dataset(tmp_path: Path, num_episodes: int, with_physics: bool) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for episode_id in range(num_episodes):
        save_trajectory_npz(
            _make_episode(seed=episode_id, with_physics=with_physics),
            dataset_dir / f"episode_{episode_id:04d}.npz",
        )
    return dataset_dir


def _save_checkpoint(tmp_path: Path, *, with_physics_in_config: bool = True) -> Path:
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
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "epoch": 0,
            "global_step": 0,
            "model_state": model.state_dict(),
            "config": {
                "q_dim": 2,
                "nuisance_dim": 4,
                "embedding_size": 32,
                "encoder_head_hidden_size": 16,
                "mass_matrix_hidden_size": 16,
                "potential_hidden_size": 16,
                "dissipation_hidden_size": 16,
                "actuation_hidden_size": 16,
                "dynamics_hidden_layers": 1,
                "dt": 0.02,
                "n_substeps": 1,
                "with_dissipation": True,
                "nuisance_alpha_init": 0.95,
                "reward_head": False,
                "reward_head_hidden_size": 200,
                "sequence_length": 32,
                "val_fraction": 0.5,
                "crop_config": None,
            },
            "model_kind": "mechanics",
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_eval_mechanics_writes_metrics_and_visualizations(tmp_path: Path) -> None:
    dataset_dir = _save_dataset(tmp_path, num_episodes=4, with_physics=True)
    checkpoint_path = _save_checkpoint(tmp_path)
    output_dir = tmp_path / "eval"
    config = MechanicsEvalConfig(
        checkpoint_path=checkpoint_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        sequence_length=16,
        batch_size=2,
        num_sequences=4,
        warmup_length=3,
        horizons=(1, 5, 10),
        val_fraction=0.5,
        num_visualizations=2,
        device="cpu",
    )
    metrics = evaluate_checkpoint(config)

    metrics_path = output_dir / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert "metrics" in payload and "config" in payload
    written = payload["metrics"]
    for key in (
        "reconstruction_mse",
        "foreground_reconstruction_mse",
        "open_loop_future_mse",
        "foreground_open_loop_future_mse",
        "qdot_smoothness_mse",
        "open_loop_mse_h1",
        "foreground_open_loop_mse_h10",
    ):
        assert key in written, f"missing metric {key!r}"
    assert written["physics_in_dataset"] is True
    assert "linear_probe" in written
    probe = written["linear_probe"]
    assert "q_to_qpos_r2_overall" in probe and "qdot_to_qvel_r2_overall" in probe
    assert "energy" in written
    assert written["energy"]["horizons_evaluated"] >= 1

    # Two MP4 + sheet pairs were requested.
    mp4s = sorted(output_dir.glob("sequence_*.mp4"))
    sheets = sorted(output_dir.glob("sequence_*_sheet.png"))
    assert len(mp4s) == 2
    assert len(sheets) == 2

    # The dictionary returned in-memory should agree with the JSON dump.
    assert metrics["reconstruction_mse"] == pytest.approx(written["reconstruction_mse"])


def test_eval_mechanics_skips_probes_when_no_physics(tmp_path: Path) -> None:
    """Datasets without ``qpos``/``qvel`` should still produce MSE metrics."""

    dataset_dir = _save_dataset(tmp_path, num_episodes=3, with_physics=False)
    checkpoint_path = _save_checkpoint(tmp_path)
    output_dir = tmp_path / "eval"
    config = MechanicsEvalConfig(
        checkpoint_path=checkpoint_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        sequence_length=16,
        batch_size=2,
        num_sequences=3,
        warmup_length=2,
        horizons=(1, 5),
        val_fraction=0.5,
        num_visualizations=1,
        device="cpu",
    )
    metrics = evaluate_checkpoint(config)
    assert metrics["physics_in_dataset"] is False
    assert "linear_probe" not in metrics
    assert "energy" in metrics  # energy uses encoded q only, still computable
