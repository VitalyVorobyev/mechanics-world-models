"""Distracting-Control-lite eval-time visual perturbations.

A thin wrapper around ``DmControlPixelEnv`` that applies per-episode color
gain and brightness to the rendered RGB frames. This is the v1 visual OOD
shift used to test research hypothesis H2 (factored latent vs entangled
under visual nuisance). Camera-pose jitter and background swap are planned
follow-ups; both are dm_control-specific and we keep this module narrow
until they're actually needed.

Eval-time only — training data stays clean. The wrapper preserves the
underlying physics (rewards, done, qpos/qvel) so the same trajectory is
playable through the perturbed pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from envs.dm_control_pixels import DmControlPixelEnv, PixelStep


@dataclass(frozen=True)
class VisualDistractorConfig:
    """Per-episode visual perturbation ranges.

    Each episode samples one ``color_gain ∈ [color_gain_low, color_gain_high]``
    per channel and one ``brightness_offset ∈ [brightness_low, brightness_high]``
    from a deterministic RNG seeded by ``seed``. Set both ``low == high`` to
    pin a perturbation to a fixed value (useful for reproducible eval).
    """

    color_gain_low: float = 1.0
    color_gain_high: float = 1.0
    brightness_low: float = 0.0
    brightness_high: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.color_gain_low <= 0.0 or self.color_gain_high <= 0.0:
            raise ValueError("color gain bounds must be positive (multiplicative)")
        if self.color_gain_high < self.color_gain_low:
            raise ValueError("color_gain_high must be >= color_gain_low")
        if self.brightness_high < self.brightness_low:
            raise ValueError("brightness_high must be >= brightness_low")


class EvalPixelEnv:
    """Image-level visual distractor wrapper around ``DmControlPixelEnv``.

    Acts as a near-drop-in for ``DmControlPixelEnv``: forwards action_spec,
    sample_action, physics_state, physics_params, etc. The wrapper resamples
    its perturbation parameters at every ``reset`` so each evaluation episode
    sees a fresh visual nuisance configuration drawn from the configured
    range. The same parameters are then held constant across the episode.
    """

    def __init__(self, base_env: DmControlPixelEnv, config: VisualDistractorConfig) -> None:
        self._base = base_env
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        self._color_gain = np.ones(3, dtype=np.float32)
        self._brightness = np.float32(0.0)

    @property
    def base_env(self) -> DmControlPixelEnv:
        """The wrapped clean env. Useful when callers need raw images."""

        return self._base

    @property
    def action_spec(self) -> Any:
        return self._base.action_spec

    @property
    def current_color_gain(self) -> NDArray[np.float32]:
        """Color gain currently applied to rendered frames (frozen across the episode)."""

        return self._color_gain.copy()

    @property
    def current_brightness(self) -> float:
        """Brightness offset currently applied to rendered frames."""

        return float(self._brightness)

    def sample_action(self, rng: np.random.Generator) -> NDArray[np.floating[Any]]:
        return self._base.sample_action(rng)

    def reset(self) -> NDArray[np.uint8]:
        self._resample_distractors()
        return self._perturb(self._base.reset())

    def step(self, action: NDArray[np.floating[Any]]) -> PixelStep:
        raw = self._base.step(action)
        return PixelStep(
            image=self._perturb(raw.image),
            reward=raw.reward,
            discount=raw.discount,
            done=raw.done,
            qpos=raw.qpos,
            qvel=raw.qvel,
        )

    def render(self) -> NDArray[np.uint8]:
        return self._perturb(self._base.render())

    def physics_state(self) -> dict[str, NDArray[np.float64]]:
        return self._base.physics_state()

    def physics_params(self) -> dict[str, float]:
        return self._base.physics_params()

    def reset_physics_state(self) -> dict[str, NDArray[np.float64]]:
        return self._base.reset_physics_state()

    def close(self) -> None:
        self._base.close()

    def __enter__(self) -> "EvalPixelEnv":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _resample_distractors(self) -> None:
        if self._config.color_gain_high > self._config.color_gain_low:
            self._color_gain = self._rng.uniform(
                self._config.color_gain_low,
                self._config.color_gain_high,
                size=3,
            ).astype(np.float32)
        else:
            self._color_gain = np.full(3, self._config.color_gain_low, dtype=np.float32)
        if self._config.brightness_high > self._config.brightness_low:
            self._brightness = np.float32(
                self._rng.uniform(self._config.brightness_low, self._config.brightness_high)
            )
        else:
            self._brightness = np.float32(self._config.brightness_low)

    def _perturb(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got dtype {image.dtype}")
        normalized = image.astype(np.float32) / 255.0
        adjusted = normalized * self._color_gain.reshape(1, 1, 3) + self._brightness
        clipped = np.clip(adjusted, 0.0, 1.0)
        return (clipped * 255.0).astype(np.uint8)
