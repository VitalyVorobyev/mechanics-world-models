"""Interrupt / resume tests for the mechanics trainer and its helpers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import train.train_mechanics as train_mechanics
from data.trajectory import Trajectory, save_trajectory_npz
from train.resumable import EpochResumableSampler, truncate_history
from train.train_mechanics import MechanicsTrainConfig, train


def _write_dataset(root: Path, num_episodes: int = 2, num_steps: int = 12) -> Path:
    dataset_dir = root / "dataset"
    dataset_dir.mkdir()
    for episode_id in range(num_episodes):
        rng = np.random.default_rng(episode_id)
        save_trajectory_npz(
            Trajectory(
                env_name="cartpole-swingup",
                seed=episode_id,
                episode_id=episode_id,
                action_repeat=2,
                image_size=84,
                images=rng.integers(0, 255, size=(num_steps + 1, 84, 84, 3)).astype(np.uint8),
                actions=rng.uniform(-1, 1, size=(num_steps, 1)).astype(np.float32),
                rewards=rng.uniform(0, 1, size=(num_steps,)).astype(np.float32),
                discounts=np.ones(num_steps, dtype=np.float32),
                dones=np.zeros(num_steps, dtype=np.bool_),
                step_indices=np.arange(num_steps, dtype=np.int64),
                frame_dt=0.02,
            ),
            dataset_dir / f"episode_{episode_id:06d}.npz",
        )
    return dataset_dir


def _config(dataset_dir: Path, checkpoint_dir: Path) -> MechanicsTrainConfig:
    return MechanicsTrainConfig(
        dataset_dir=dataset_dir,
        checkpoint_dir=checkpoint_dir,
        sequence_length=6,
        batch_size=2,
        epochs=2,
        nuisance_dim=4,
        embedding_size=32,
        encoder_head_hidden_size=16,
        mass_matrix_hidden_size=16,
        potential_hidden_size=16,
        dissipation_hidden_size=16,
        actuation_hidden_size=16,
        dynamics_hidden_layers=1,
        imagination_context_steps=3,
        imagination_horizon=3,
        angle_dims=(1,),
        nuisance_mode="style",
        qdot_stencil="backward",
        dissipation_form="linear",
        val_fraction=0.5,
        device="cpu",
        log_every_steps=1,
        warmup_steps=1,
    )


class _StopAfter:
    """Stand-in for ``GracefulStop`` that requests a stop after N step checks."""

    def __init__(self, checks_before_stop: int) -> None:
        self._remaining = checks_before_stop

    @property
    def requested(self) -> bool:
        self._remaining -= 1
        return self._remaining < 0

    def __enter__(self) -> "_StopAfter":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _final_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    return torch.load(checkpoint_dir / "latest.pt", weights_only=False)["model_state"]


def test_sampler_skip_resumes_the_same_permutation() -> None:
    sampler = EpochResumableSampler(10, seed=3)
    sampler.set_epoch(2)
    full = list(sampler)
    sampler.set_epoch(2, skip_samples=4)
    assert list(sampler) == full[4:]
    assert len(sampler) == 6
    sampler.set_epoch(3)
    assert list(sampler) != full  # a different epoch reshuffles


def test_truncate_history_drops_rows_after_step(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    rows = [{"global_step": s, "split": "train"} for s in (1, 2, 3, 4)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"global_st')
    assert truncate_history(path, 2) == 3  # steps 3, 4 and the cut-off line
    assert [json.loads(l)["global_step"] for l in path.read_text().splitlines()] == [1, 2]


def test_interrupted_run_resumes_to_identical_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = _write_dataset(tmp_path)

    reference_dir = tmp_path / "reference"
    train(_config(dataset_dir, reference_dir))
    reference = _final_state(reference_dir)

    resumed_dir = tmp_path / "resumed"
    config = _config(dataset_dir, resumed_dir)
    # Stop in the middle of epoch 1 (each training step checks the flag once).
    monkeypatch.setattr(train_mechanics, "GracefulStop", lambda: _StopAfter(1))
    train(config)
    progress = json.loads((resumed_dir / "progress.json").read_text())
    assert progress["status"] == "stopped"
    partial = torch.load(resumed_dir / "latest.pt", weights_only=False)["partial_epoch"]
    assert partial == {**partial, "epoch": 1, "batches_done": 2}

    monkeypatch.setattr(train_mechanics, "GracefulStop", lambda: _StopAfter(10_000))
    train(replace(config, auto_resume=True))

    resumed = _final_state(resumed_dir)
    for name, tensor in reference.items():
        assert torch.equal(tensor, resumed[name]), name
    steps = [
        json.loads(line)["global_step"]
        for line in (resumed_dir / "history.jsonl").read_text().splitlines()
        if json.loads(line)["split"] == "train"
    ]
    assert steps == sorted(set(steps)), "history has duplicated or out-of-order steps"
    assert json.loads((resumed_dir / "progress.json").read_text())["status"] == "finished"
