from __future__ import annotations

import numpy as np
import pytest

from data import Trajectory, load_trajectory_npz, save_trajectory_npz


def test_trajectory_npz_round_trip(tmp_path) -> None:
    trajectory = Trajectory(
        env_name="cartpole-swingup",
        seed=7,
        episode_id=3,
        action_repeat=2,
        image_size=84,
        images=np.zeros((3, 84, 84, 3), dtype=np.uint8),
        actions=np.asarray([[0.1], [-0.2]], dtype=np.float32),
        rewards=np.asarray([1.0, 0.5], dtype=np.float32),
        discounts=np.asarray([1.0, 0.0], dtype=np.float32),
        dones=np.asarray([False, True], dtype=np.bool_),
        step_indices=np.asarray([0, 1], dtype=np.int64),
    )

    path = save_trajectory_npz(trajectory, tmp_path / "episode_000003.npz")
    loaded = load_trajectory_npz(path)

    assert loaded.env_name == trajectory.env_name
    assert loaded.seed == trajectory.seed
    assert loaded.episode_id == trajectory.episode_id
    assert loaded.action_repeat == trajectory.action_repeat
    assert loaded.image_size == trajectory.image_size
    np.testing.assert_array_equal(loaded.images, trajectory.images)
    np.testing.assert_array_equal(loaded.actions, trajectory.actions)
    np.testing.assert_array_equal(loaded.rewards, trajectory.rewards)
    np.testing.assert_array_equal(loaded.discounts, trajectory.discounts)
    np.testing.assert_array_equal(loaded.dones, trajectory.dones)
    np.testing.assert_array_equal(loaded.step_indices, trajectory.step_indices)
    # Legacy schema: physics fields absent, loader returns None.
    assert loaded.qpos is None
    assert loaded.qvel is None
    assert loaded.physics_params is None


def test_trajectory_npz_round_trip_with_physics(tmp_path) -> None:
    qpos = np.asarray([[0.0, 0.1], [0.05, 0.2], [0.10, 0.3]], dtype=np.float64)
    qvel = np.asarray([[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]], dtype=np.float64)
    physics_params = {"body_mass_1": 1.0, "body_mass_2": 0.1, "dof_damping_0": 0.0}
    trajectory = Trajectory(
        env_name="cartpole-swingup",
        seed=7,
        episode_id=3,
        action_repeat=2,
        image_size=84,
        images=np.zeros((3, 84, 84, 3), dtype=np.uint8),
        actions=np.asarray([[0.1], [-0.2]], dtype=np.float32),
        rewards=np.asarray([1.0, 0.5], dtype=np.float32),
        discounts=np.asarray([1.0, 0.0], dtype=np.float32),
        dones=np.asarray([False, True], dtype=np.bool_),
        step_indices=np.asarray([0, 1], dtype=np.int64),
        qpos=qpos,
        qvel=qvel,
        physics_params=physics_params,
    )

    path = save_trajectory_npz(trajectory, tmp_path / "episode_000003.npz")
    loaded = load_trajectory_npz(path)

    assert loaded.qpos is not None and loaded.qvel is not None
    np.testing.assert_array_equal(loaded.qpos, qpos)
    np.testing.assert_array_equal(loaded.qvel, qvel)
    assert loaded.physics_params == physics_params


def test_trajectory_validates_physics_alignment() -> None:
    bad_qpos = np.zeros((2, 2), dtype=np.float64)  # T+1 should be 3
    bad_qvel = np.zeros((3, 2), dtype=np.float64)
    trajectory = Trajectory(
        env_name="cartpole-swingup",
        seed=0,
        episode_id=0,
        action_repeat=1,
        image_size=84,
        images=np.zeros((3, 84, 84, 3), dtype=np.uint8),
        actions=np.zeros((2, 1), dtype=np.float32),
        rewards=np.zeros(2, dtype=np.float32),
        discounts=np.ones(2, dtype=np.float32),
        dones=np.asarray([False, True], dtype=np.bool_),
        step_indices=np.asarray([0, 1], dtype=np.int64),
        qpos=bad_qpos,
        qvel=bad_qvel,
    )
    with pytest.raises(ValueError, match="qpos must contain one entry per image frame"):
        trajectory.validate()
