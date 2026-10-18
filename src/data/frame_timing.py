"""Resolve the simulated time between consecutive frames of a dataset.

Continuous-time world models (the mechanics model's integrator) need ``dt`` in
seconds per *frame*, which is the suite's control timestep times the
collection ``action_repeat`` — not the raw physics timestep. Datasets written
after the frame-timing field was added store it directly; older datasets only
store ``env_name`` and ``action_repeat``, so we ask dm_control for the control
timestep of that environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.sequence_dataset import index_episode_files


def resolve_dataset_frame_dt(dataset_dir: Path) -> tuple[float, str]:
    """Return ``(frame_dt_seconds, source)`` for the first episode in ``dataset_dir``.

    ``source`` is a short human-readable note on where the value came from so
    trainers can log it next to the resolved ``dt``.
    """

    episode_files = index_episode_files(dataset_dir)
    if not episode_files:
        raise ValueError(f"no episode files found in {dataset_dir}")
    with np.load(episode_files[0], allow_pickle=False) as data:
        if "frame_dt" in data.files:
            return float(data["frame_dt"].item()), "dataset metadata (frame_dt)"
        env_name = str(data["env_name"].item())
        action_repeat = int(data["action_repeat"].item())

    from dm_control import suite

    domain_name, task_name = env_name.split("-", maxsplit=1)
    env = suite.load(domain_name=domain_name, task_name=task_name)
    try:
        control_timestep = float(env.control_timestep())
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
    return (
        control_timestep * action_repeat,
        f"dm_control control_timestep {control_timestep:g} x action_repeat {action_repeat}",
    )
