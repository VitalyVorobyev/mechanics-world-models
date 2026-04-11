from __future__ import annotations

import numpy as np

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
