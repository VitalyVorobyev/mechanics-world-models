"""Pixel observation wrapper for DeepMind Control Suite environments."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from dm_control import suite
from numpy.typing import NDArray


_BODY_MASS_RE = re.compile(r"^body_mass_(\d+)$")
_DOF_DAMPING_RE = re.compile(r"^dof_damping_(\d+)$")
_GEOM_SIZE_RE = re.compile(r"^geom_size_(\d+)_(\d+)$")


@dataclass(frozen=True)
class PixelEnvConfig:
    """Configuration for a dm_control pixel environment.

    ``physics_param_scales`` is a flat dict whose keys mirror the ones
    returned by ``DmControlPixelEnv.physics_params`` (``body_mass_<i>``,
    ``dof_damping_<i>``, ``geom_size_<i>_<j>``). Each value is a positive
    multiplicative scale applied immediately after the env is loaded — used
    for OOD physics shifts (mass×k, length×k, damping×k). ``None`` keeps the
    nominal physics.
    """

    env_name: str = "cartpole-swingup"
    seed: int = 0
    action_repeat: int = 1
    image_size: int = 84
    camera_id: int | str = 0
    physics_param_scales: dict[str, float] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")
        if self.image_size < 1:
            raise ValueError("image_size must be >= 1")
        if "-" not in self.env_name:
            raise ValueError("env_name must use '<domain>-<task>', e.g. 'cartpole-swingup'")
        if self.physics_param_scales is not None:
            for key, scale in self.physics_param_scales.items():
                if scale <= 0.0:
                    raise ValueError(
                        f"physics_param_scales[{key!r}]={scale} must be positive",
                    )
                if not (
                    _BODY_MASS_RE.match(key)
                    or _DOF_DAMPING_RE.match(key)
                    or _GEOM_SIZE_RE.match(key)
                ):
                    raise ValueError(
                        f"unknown physics_param_scales key {key!r}; expected "
                        "body_mass_<i>, dof_damping_<i>, or geom_size_<i>_<j>",
                    )


@dataclass(frozen=True)
class PixelStep:
    """One action-repeat step with the rendered next observation."""

    image: NDArray[np.uint8]
    reward: float
    discount: float
    done: bool
    qpos: NDArray[np.float64] | None = None
    qvel: NDArray[np.float64] | None = None


class DmControlPixelEnv:
    """Small wrapper around dm_control that exposes RGB image observations."""

    def __init__(self, config: PixelEnvConfig) -> None:
        self.config = config
        domain_name, task_name = config.env_name.split("-", maxsplit=1)
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={"random": config.seed},
        )
        if config.physics_param_scales:
            apply_physics_param_scales(self._env.physics, config.physics_param_scales)
        self._time_step: Any | None = None

    @property
    def action_spec(self) -> Any:
        """Return the wrapped dm_control action spec."""

        return self._env.action_spec()

    def sample_action(self, rng: np.random.Generator) -> NDArray[np.floating[Any]]:
        """Sample a uniformly random action inside the action spec bounds."""

        spec = self.action_spec
        return np.asarray(
            rng.uniform(low=spec.minimum, high=spec.maximum),
            dtype=spec.dtype,
        )

    def reset(self) -> NDArray[np.uint8]:
        """Reset the environment and return the initial RGB observation."""

        self._time_step = self._env.reset()
        return self.render()

    def step(self, action: NDArray[np.floating[Any]]) -> PixelStep:
        """Apply one action for ``action_repeat`` physics steps."""

        reward = 0.0
        discount = 1.0
        done = False

        for _ in range(self.config.action_repeat):
            self._time_step = self._env.step(action)
            reward += float(self._time_step.reward or 0.0)
            if self._time_step.discount is not None:
                discount = float(self._time_step.discount)
            done = bool(self._time_step.last())
            if done:
                break

        physics_state = self.physics_state()
        return PixelStep(
            image=self.render(),
            reward=reward,
            discount=discount,
            done=done,
            qpos=physics_state["qpos"],
            qvel=physics_state["qvel"],
        )

    def physics_state(self) -> dict[str, NDArray[np.float64]]:
        """Return a snapshot of the simulator's generalized state.

        Returns ``{"qpos": ..., "qvel": ...}`` as float64 copies so callers can
        accumulate them across the rollout without aliasing MuJoCo's buffers.
        """

        physics = self._env.physics
        return {
            "qpos": np.asarray(physics.data.qpos, dtype=np.float64).copy(),
            "qvel": np.asarray(physics.data.qvel, dtype=np.float64).copy(),
        }

    def physics_params(self) -> dict[str, float]:
        """Snapshot of static physical parameters (masses, dampings, geom sizes).

        These are constant across the episode (we don't re-load mid-run) and
        identify which OOD shift, if any, was applied. Stored as a flat dict
        of named scalars so the JSON dump in the trajectory NPZ stays simple.
        """

        physics = self._env.physics
        params: dict[str, float] = {}
        body_mass = np.asarray(physics.model.body_mass, dtype=np.float64)
        for i, mass in enumerate(body_mass):
            params[f"body_mass_{i}"] = float(mass)
        dof_damping = np.asarray(physics.model.dof_damping, dtype=np.float64)
        for i, damping in enumerate(dof_damping):
            params[f"dof_damping_{i}"] = float(damping)
        geom_size = np.asarray(physics.model.geom_size, dtype=np.float64)
        for i in range(geom_size.shape[0]):
            for j in range(geom_size.shape[1]):
                params[f"geom_size_{i}_{j}"] = float(geom_size[i, j])
        return params

    def reset_physics_state(self) -> dict[str, NDArray[np.float64]]:
        """Return the physics state right after ``reset`` so callers can store ``qpos_0``."""

        return self.physics_state()

    def render(self) -> NDArray[np.uint8]:
        """Render the current simulator state as an RGB uint8 image."""

        image = self._env.physics.render(
            height=self.config.image_size,
            width=self.config.image_size,
            camera_id=self.config.camera_id,
        )
        return np.asarray(image, dtype=np.uint8)

    def close(self) -> None:
        """Release resources held by the underlying environment."""

        close = getattr(self._env, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "DmControlPixelEnv":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def apply_physics_param_scales(physics: Any, scales: dict[str, float]) -> None:
    """Mutate ``physics.model`` arrays in place by the given multiplicative scales.

    Designed to be reusable for both data collection (perturbed datasets) and
    eval-time OOD probes. Keys use the same names as
    ``DmControlPixelEnv.physics_params`` so the perturbation grid in scripts
    matches the JSON payload stored in the .npz.
    """

    for key, scale in scales.items():
        if scale <= 0.0:
            raise ValueError(f"physics scale {key!r}={scale} must be positive")
        if (match := _BODY_MASS_RE.match(key)) is not None:
            index = int(match.group(1))
            _scale_array_entry(physics.model.body_mass, (index,), scale, key)
        elif (match := _DOF_DAMPING_RE.match(key)) is not None:
            index = int(match.group(1))
            _scale_array_entry(physics.model.dof_damping, (index,), scale, key)
        elif (match := _GEOM_SIZE_RE.match(key)) is not None:
            i = int(match.group(1))
            j = int(match.group(2))
            _scale_array_entry(physics.model.geom_size, (i, j), scale, key)
        else:
            raise ValueError(
                f"unknown physics scale key {key!r}; expected body_mass_<i>, "
                "dof_damping_<i>, or geom_size_<i>_<j>",
            )


def _scale_array_entry(
    array: Any,
    indices: tuple[int, ...],
    scale: float,
    key: str,
) -> None:
    """Multiply ``array[indices]`` by ``scale``, raising a clear error on shape mismatch."""

    try:
        original = float(array[indices])
    except IndexError as exc:
        raise IndexError(
            f"physics scale {key!r} indexes {indices} into an array of shape {array.shape}",
        ) from exc
    array[indices] = np.float64(original * scale)
