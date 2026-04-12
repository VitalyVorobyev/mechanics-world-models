from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval import eval_rssm as eval_rssm_module
from data import Trajectory, save_trajectory_npz
from eval.dataset_diagnostics import DatasetImageDiagnosticsConfig, run_dataset_image_diagnostics
from eval.debug_reconstruction import ReconstructionDebugConfig, run_reconstruction_debug
from eval.eval_rssm import (
    EvalConfig,
    EvalRollout,
    deterministic_eval_rollout,
    evaluate_checkpoint,
    load_model_from_checkpoint,
)
from eval.foreground_masks import ForegroundMaskPreviewConfig, run_foreground_mask_preview
from eval.overfit_reconstruction import OverfitReconstructionConfig, run_overfit_reconstruction
from models import VisualWorldModel
from train.train_rssm import serializable_config
from train.train_rssm import TrainConfig
from viz.rssm_eval_viz import write_prediction_contact_sheet, write_prediction_video


def _observations(batch_size: int = 2, sequence_length: int = 4) -> torch.Tensor:
    return torch.rand(batch_size, sequence_length, 3, 84, 84)


def _actions(batch_size: int = 2, sequence_length: int = 4, action_dim: int = 1) -> torch.Tensor:
    return torch.rand(batch_size, sequence_length, action_dim)


def _write_episode(root: Path, episode_id: int, num_steps: int = 4) -> Path:
    rng = np.random.default_rng(episode_id)
    trajectory = Trajectory(
        env_name="cartpole-swingup",
        seed=0,
        episode_id=episode_id,
        action_repeat=2,
        image_size=84,
        images=rng.integers(0, 256, size=(num_steps + 1, 84, 84, 3), dtype=np.uint8),
        actions=rng.uniform(-1.0, 1.0, size=(num_steps, 1)).astype(np.float32),
        rewards=rng.normal(size=num_steps).astype(np.float32),
        discounts=np.ones(num_steps, dtype=np.float32),
        dones=np.asarray([False] * (num_steps - 1) + [True], dtype=np.bool_),
        step_indices=np.arange(num_steps, dtype=np.int64),
    )
    return save_trajectory_npz(trajectory, root / f"episode_{episode_id:06d}.npz")


def _write_checkpoint(path: Path) -> Path:
    model = VisualWorldModel(action_dim=1, embedding_size=32, hidden_size=16, latent_size=8)
    config = TrainConfig(
        dataset_dir=path.parent,
        checkpoint_dir=path.parent,
        sequence_length=3,
        batch_size=2,
        hidden_size=16,
        latent_size=8,
        embedding_size=32,
        val_fraction=0.5,
    )
    payload = {
        "epoch": 1,
        "global_step": 1,
        "model_state": model.state_dict(),
        "optimizer_state": {},
        "config": serializable_config(config),
        "metrics": {},
        "history_path": str(path.parent / "history.jsonl"),
    }
    torch.save(payload, path)
    return path


def test_deterministic_eval_rollout_shapes() -> None:
    model = VisualWorldModel(action_dim=1, embedding_size=32, hidden_size=16, latent_size=8)
    outputs = deterministic_eval_rollout(
        model=model,
        observations=_observations(),
        actions=_actions(),
        warmup_length=2,
    )

    assert outputs.reconstructions.shape == (2, 4, 3, 84, 84)
    assert outputs.one_step_predictions.shape == (2, 4, 3, 84, 84)
    assert outputs.open_loop_predictions.shape == (2, 4, 3, 84, 84)
    assert torch.all(outputs.reconstructions >= 0.0)
    assert torch.all(outputs.reconstructions <= 1.0)


def test_legacy_checkpoint_without_latent_predictor_is_tolerated() -> None:
    model = VisualWorldModel(action_dim=1, embedding_size=32, hidden_size=16, latent_size=8)
    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("latent_predictor.")
    }

    loaded = load_model_from_checkpoint(
        checkpoint={"model_state": legacy_state},
        checkpoint_config={"embedding_size": 32, "hidden_size": 16, "latent_size": 8},
        action_dim=1,
        image_size=84,
        device=torch.device("cpu"),
    )

    assert getattr(loaded, "latent_predictor_loaded") is False


def test_evaluate_checkpoint_writes_metrics(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    for episode_id in range(4):
        _write_episode(dataset_dir, episode_id)
    checkpoint_path = _write_checkpoint(tmp_path / "latest.pt")
    output_dir = tmp_path / "eval"

    metrics = evaluate_checkpoint(
        EvalConfig(
            checkpoint_path=checkpoint_path,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            sequence_length=3,
            batch_size=2,
            num_sequences=2,
            warmup_length=1,
            horizons=(1, 2),
            seed=0,
            val_fraction=0.5,
            num_visualizations=0,
            device="cpu",
        ),
    )

    assert (output_dir / "metrics.json").exists()
    assert set(metrics) >= {
        "reconstruction_mse",
        "one_step_prediction_mse",
        "foreground_one_step_prediction_mse",
        "open_loop_future_mse",
        "foreground_open_loop_future_mse",
        "open_loop_mse_h1",
        "open_loop_mse_h2",
        "foreground_open_loop_mse_h1",
        "foreground_open_loop_mse_h2",
        "latent_consistency_loss",
    }
    for value in metrics.values():
        assert np.isfinite(value)


def test_evaluate_loader_compares_predictions_to_next_observations(monkeypatch, tmp_path) -> None:
    sequence_length = 4
    sample = {
        "observations": torch.zeros(sequence_length, 3, 4, 4),
        "next_observations": torch.ones(sequence_length, 3, 4, 4),
        "actions": torch.zeros(sequence_length, 1),
    }
    loader = DataLoader([sample], batch_size=1)

    def fake_rollout(
        model: VisualWorldModel,
        observations: torch.Tensor,
        actions: torch.Tensor,
        warmup_length: int,
    ) -> EvalRollout:
        return EvalRollout(
            reconstructions=observations.clone(),
            one_step_predictions=torch.ones_like(observations),
            open_loop_predictions=torch.ones_like(observations),
        )

    model = VisualWorldModel(action_dim=1, embedding_size=32, hidden_size=16, latent_size=8)
    setattr(model, "latent_predictor_loaded", False)
    monkeypatch.setattr(eval_rssm_module, "deterministic_eval_rollout", fake_rollout)

    metrics = eval_rssm_module.evaluate_loader(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        warmup_length=2,
        horizons=(1, 2),
        output_dir=tmp_path,
        num_visualizations=0,
        video_fps=2,
        sheet_timesteps=2,
    )

    assert metrics["reconstruction_mse"] == 0.0
    assert metrics["one_step_prediction_mse"] == 0.0
    assert metrics["foreground_one_step_prediction_mse"] == 0.0
    assert metrics["open_loop_future_mse"] == 0.0
    assert metrics["foreground_open_loop_future_mse"] == 0.0
    assert metrics["open_loop_mse_h1"] == 0.0
    assert metrics["foreground_open_loop_mse_h1"] == 0.0


def test_open_loop_horizon_one_targets_first_frame_after_warmup(monkeypatch, tmp_path) -> None:
    sequence_length = 4
    observations = torch.zeros(sequence_length, 3, 4, 4)
    next_observations = torch.stack(
        [torch.full((3, 4, 4), float(timestep)) for timestep in range(sequence_length)],
        dim=0,
    )
    sample = {
        "observations": observations,
        "next_observations": next_observations,
        "actions": torch.zeros(sequence_length, 1),
    }
    loader = DataLoader([sample], batch_size=1)

    def fake_rollout(
        model: VisualWorldModel,
        observations: torch.Tensor,
        actions: torch.Tensor,
        warmup_length: int,
    ) -> EvalRollout:
        open_loop_predictions = torch.full_like(observations, -10.0)
        open_loop_predictions[:, warmup_length - 1] = next_observations[warmup_length - 1]
        return EvalRollout(
            reconstructions=observations.clone(),
            one_step_predictions=torch.zeros_like(observations),
            open_loop_predictions=open_loop_predictions,
        )

    model = VisualWorldModel(action_dim=1, embedding_size=32, hidden_size=16, latent_size=8)
    setattr(model, "latent_predictor_loaded", False)
    monkeypatch.setattr(eval_rssm_module, "deterministic_eval_rollout", fake_rollout)

    metrics = eval_rssm_module.evaluate_loader(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        warmup_length=2,
        horizons=(1,),
        output_dir=tmp_path,
        num_visualizations=0,
        video_fps=2,
        sheet_timesteps=2,
    )

    assert metrics["open_loop_mse_h1"] == 0.0
    assert metrics["foreground_open_loop_mse_h1"] == 0.0
    assert metrics["open_loop_future_mse"] > 0.0


def test_prediction_visualization_writes_artifacts(tmp_path) -> None:
    ground_truth = np.zeros((3, 3, 4, 4), dtype=np.float32)
    reconstruction = np.full((3, 3, 4, 4), 0.5, dtype=np.float32)
    prediction = np.ones((3, 3, 4, 4), dtype=np.float32)

    video_path = write_prediction_video(
        output_path=tmp_path / "prediction.mp4",
        ground_truth=ground_truth,
        reconstruction=reconstruction,
        open_loop_prediction=prediction,
        fps=2,
    )
    sheet_path = write_prediction_contact_sheet(
        output_path=tmp_path / "prediction_sheet.png",
        ground_truth=ground_truth,
        reconstruction=reconstruction,
        open_loop_prediction=prediction,
        timesteps=[0, 2],
    )

    assert video_path.exists()
    assert video_path.stat().st_size > 0
    assert sheet_path.exists()
    assert sheet_path.stat().st_size > 0


def test_reconstruction_debug_writes_train_val_grids(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    for episode_id in range(4):
        _write_episode(dataset_dir, episode_id)
    checkpoint_path = _write_checkpoint(tmp_path / "latest.pt")
    output_dir = tmp_path / "debug_reconstruction"

    results = run_reconstruction_debug(
        ReconstructionDebugConfig(
            checkpoint_path=checkpoint_path,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            sequence_length=3,
            batch_size=1,
            timesteps=(0, 2),
            val_fraction=0.5,
            device="cpu",
        ),
    )

    assert (output_dir / "train_posterior_reconstruction_grid.png").exists()
    assert (output_dir / "val_posterior_reconstruction_grid.png").exists()
    assert (output_dir / "reconstruction_debug.json").exists()
    assert set(results["splits"]) == {"train", "val"}
    assert np.isfinite(results["splits"]["train"]["mse"])


def test_dataset_image_diagnostics_writes_average_and_variance(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    for episode_id in range(2):
        _write_episode(dataset_dir, episode_id)
    output_dir = tmp_path / "dataset_images"

    stats = run_dataset_image_diagnostics(
        DatasetImageDiagnosticsConfig(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            max_frames=5,
            seed=0,
        ),
    )

    assert (output_dir / "average_image.png").exists()
    assert (output_dir / "variance_image.png").exists()
    assert (output_dir / "variance_heatmap.png").exists()
    assert stats["sampled_frames"] == 5
    assert stats["total_frames_seen"] == 10


def test_foreground_mask_preview_writes_contact_sheet(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    for episode_id in range(4):
        _write_episode(dataset_dir, episode_id)
    output_path = tmp_path / "foreground_masks.png"

    metadata = run_foreground_mask_preview(
        ForegroundMaskPreviewConfig(
            dataset_dir=dataset_dir,
            output_path=output_path,
            sequence_length=3,
            batch_size=2,
            timesteps=(0, 2),
            val_fraction=0.5,
            seed=0,
        ),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.with_suffix(".json").exists()
    assert metadata["output_path"] == str(output_path)
    assert metadata["timesteps"] == [0, 2]


def test_overfit_reconstruction_smoke_writes_grid(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    for episode_id in range(4):
        _write_episode(dataset_dir, episode_id)
    checkpoint_path = _write_checkpoint(tmp_path / "latest.pt")
    output_dir = tmp_path / "overfit"

    summary = run_overfit_reconstruction(
        OverfitReconstructionConfig(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            sequence_length=3,
            batch_size=1,
            mode="sequence",
            steps=1,
            save_every=1,
            val_fraction=0.5,
            device="cpu",
        ),
    )

    assert (output_dir / "reconstruction_step_000000.png").exists()
    assert (output_dir / "reconstruction_step_000001.png").exists()
    assert (output_dir / "overfit_history.jsonl").exists()
    assert (output_dir / "overfit_latest.pt").exists()
    assert np.isfinite(summary["final_metrics"]["total_loss"])
