"""Pixel observation wrapper for DeepMind Control Suite environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from dm_control import suite
from numpy.typing import NDArray


@dataclass(frozen=True)
class PixelEnvConfig:
    """Configuration for a dm_control pixel environment."""

    env_name: str = "cartpole-swingup"
    seed: int = 0
    action_repeat: int = 1
    image_size: int = 84
    camera_id: int | str = 0

    def __post_init__(self) -> None:
        if self.action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")
        if self.image_size < 1:
            raise ValueError("image_size must be >= 1")
        if "-" not in self.env_name:
            raise ValueError("env_name must use '<domain>-<task>', e.g. 'cartpole-swingup'")


@dataclass(frozen=True)
class PixelStep:
    """One action-repeat step with the rendered next observation."""

    image: NDArray[np.uint8]
    reward: float
    discount: float
    done: bool


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

        return PixelStep(
            image=self.render(),
            reward=reward,
            discount=discount,
            done=done,
        )

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
