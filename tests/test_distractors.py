"""Unit tests for the visual distractor wrapper using a stub base env."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from envs.dm_control_pixels import PixelStep
from envs.distractors import EvalPixelEnv, VisualDistractorConfig


@dataclass
class _StubEnv:
    """Minimal stand-in for DmControlPixelEnv in unit tests."""

    image_size: int = 84
    reset_calls: int = 0
    step_calls: int = 0

    def reset(self) -> np.ndarray:
        self.reset_calls += 1
        # Mid-grey 0.5 * 255 = 127 across the frame.
        return np.full((self.image_size, self.image_size, 3), 127, dtype=np.uint8)

    def step(self, action: np.ndarray) -> PixelStep:
        self.step_calls += 1
        return PixelStep(
            image=np.full((self.image_size, self.image_size, 3), 127, dtype=np.uint8),
            reward=0.5,
            discount=1.0,
            done=False,
            qpos=np.zeros(2, dtype=np.float64),
            qvel=np.zeros(2, dtype=np.float64),
        )

    def render(self) -> np.ndarray:
        return np.full((self.image_size, self.image_size, 3), 127, dtype=np.uint8)

    def physics_state(self) -> dict[str, np.ndarray]:
        return {"qpos": np.zeros(2), "qvel": np.zeros(2)}

    def physics_params(self) -> dict[str, float]:
        return {"body_mass_2": 0.1}

    def reset_physics_state(self) -> dict[str, np.ndarray]:
        return self.physics_state()

    @property
    def action_spec(self) -> Any:
        return object()

    def sample_action(self, rng: np.random.Generator) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    def close(self) -> None:  # pragma: no cover
        pass


def test_distractor_wrapper_passes_through_when_ranges_collapse() -> None:
    base = _StubEnv()
    config = VisualDistractorConfig(
        color_gain_low=1.0,
        color_gain_high=1.0,
        brightness_low=0.0,
        brightness_high=0.0,
    )
    wrapper = EvalPixelEnv(base, config)
    image = wrapper.reset()
    np.testing.assert_array_equal(image, base.reset())  # identity transform
    np.testing.assert_array_equal(wrapper.current_color_gain, np.ones(3))
    assert wrapper.current_brightness == pytest.approx(0.0)


def test_distractor_wrapper_applies_color_gain_per_channel() -> None:
    base = _StubEnv()
    config = VisualDistractorConfig(
        color_gain_low=1.5,
        color_gain_high=1.5,
        brightness_low=0.0,
        brightness_high=0.0,
    )
    wrapper = EvalPixelEnv(base, config)
    image = wrapper.reset()
    # 127/255 * 1.5 = 0.7470... → 190
    expected = int(round((127 / 255.0) * 1.5 * 255.0))
    assert image[0, 0, 0] == expected
    assert image[0, 0, 1] == expected
    assert image[0, 0, 2] == expected


def test_distractor_wrapper_clips_above_one() -> None:
    base = _StubEnv()
    config = VisualDistractorConfig(
        color_gain_low=10.0,
        color_gain_high=10.0,
        brightness_low=0.0,
        brightness_high=0.0,
    )
    wrapper = EvalPixelEnv(base, config)
    image = wrapper.reset()
    assert int(image.max()) == 255
    assert int(image.min()) == 255


def test_distractor_wrapper_resamples_per_episode() -> None:
    base = _StubEnv()
    config = VisualDistractorConfig(
        color_gain_low=0.5,
        color_gain_high=2.0,
        brightness_low=-0.1,
        brightness_high=0.1,
        seed=42,
    )
    wrapper = EvalPixelEnv(base, config)
    wrapper.reset()
    first_gain = wrapper.current_color_gain
    first_brightness = wrapper.current_brightness
    # Same episode: step() does not re-sample.
    wrapper.step(np.zeros(1, dtype=np.float32))
    np.testing.assert_array_equal(wrapper.current_color_gain, first_gain)
    assert wrapper.current_brightness == pytest.approx(first_brightness)
    # New episode (reset): re-samples.
    wrapper.reset()
    second_gain = wrapper.current_color_gain
    assert not np.allclose(first_gain, second_gain)


def test_distractor_wrapper_preserves_physics_pass_through() -> None:
    base = _StubEnv()
    wrapper = EvalPixelEnv(base, VisualDistractorConfig(color_gain_low=0.5, color_gain_high=2.0))
    wrapper.reset()
    step = wrapper.step(np.zeros(1, dtype=np.float32))
    assert step.reward == 0.5
    assert step.qpos is not None and step.qvel is not None
    np.testing.assert_array_equal(step.qpos, np.zeros(2, dtype=np.float64))


def test_visual_distractor_config_validates_ranges() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        VisualDistractorConfig(color_gain_low=-1.0, color_gain_high=1.0)
    with pytest.raises(ValueError, match="color_gain_high"):
        VisualDistractorConfig(color_gain_low=2.0, color_gain_high=1.0)
    with pytest.raises(ValueError, match="brightness_high"):
        VisualDistractorConfig(brightness_low=0.5, brightness_high=0.1)
