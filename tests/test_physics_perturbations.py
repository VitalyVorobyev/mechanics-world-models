"""Unit tests for the physics perturbation wrapper.

Uses a stub physics object so the tests run without MuJoCo / dm_control.
The corresponding integration with a real env is exercised by the
``test_env_pixels`` suite, which is gated on offscreen rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from envs.dm_control_pixels import (
    PixelEnvConfig,
    apply_physics_param_scales,
)


@dataclass
class _StubModel:
    body_mass: np.ndarray
    geom_size: np.ndarray
    dof_damping: np.ndarray


@dataclass
class _StubPhysics:
    model: _StubModel


def _make_physics() -> _StubPhysics:
    return _StubPhysics(
        model=_StubModel(
            body_mass=np.asarray([0.0, 1.0, 0.1], dtype=np.float64),
            geom_size=np.asarray(
                [[0.0, 0.0, 0.0], [0.05, 0.6, 0.0], [0.045, 0.3, 0.0]],
                dtype=np.float64,
            ),
            dof_damping=np.asarray([0.1, 0.05], dtype=np.float64),
        ),
    )


def test_apply_physics_param_scales_mutates_named_entries() -> None:
    physics = _make_physics()
    apply_physics_param_scales(
        physics,
        {
            "body_mass_2": 0.5,
            "geom_size_2_1": 1.5,
            "dof_damping_1": 2.0,
        },
    )

    assert physics.model.body_mass[2] == pytest.approx(0.05)
    assert physics.model.body_mass[1] == pytest.approx(1.0)  # untouched
    assert physics.model.geom_size[2, 1] == pytest.approx(0.45)
    assert physics.model.dof_damping[1] == pytest.approx(0.10)
    assert physics.model.dof_damping[0] == pytest.approx(0.10)  # untouched


def test_apply_physics_param_scales_rejects_negative() -> None:
    physics = _make_physics()
    with pytest.raises(ValueError, match="must be positive"):
        apply_physics_param_scales(physics, {"body_mass_2": -0.5})


def test_apply_physics_param_scales_rejects_unknown_keys() -> None:
    physics = _make_physics()
    with pytest.raises(ValueError, match="unknown physics scale key"):
        apply_physics_param_scales(physics, {"pole_length": 1.5})


def test_apply_physics_param_scales_reports_index_errors() -> None:
    physics = _make_physics()
    with pytest.raises(IndexError, match="indexes"):
        apply_physics_param_scales(physics, {"body_mass_99": 0.5})


def test_pixel_env_config_validates_scales() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        PixelEnvConfig(physics_param_scales={"body_mass_2": -0.1})
    with pytest.raises(ValueError, match="unknown physics_param_scales key"):
        PixelEnvConfig(physics_param_scales={"banana": 1.0})
    # Valid spec passes.
    PixelEnvConfig(physics_param_scales={"body_mass_2": 0.5, "dof_damping_1": 2.0})
