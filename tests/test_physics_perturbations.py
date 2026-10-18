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
    body_inertia: np.ndarray
    geom_size: np.ndarray
    dof_damping: np.ndarray


@dataclass
class _StubPhysics:
    model: _StubModel


def _make_physics() -> _StubPhysics:
    return _StubPhysics(
        model=_StubModel(
            body_mass=np.asarray([0.0, 1.0, 0.1], dtype=np.float64),
            body_inertia=np.asarray(
                [[0.0, 0.0, 0.0], [0.01, 0.02, 0.03], [0.009, 0.009, 0.0001]],
                dtype=np.float64,
            ),
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
    # Mass scaling is a density change: inertia scales with it.
    np.testing.assert_allclose(physics.model.body_inertia[2], [0.0045, 0.0045, 0.00005])
    np.testing.assert_allclose(physics.model.body_inertia[1], [0.01, 0.02, 0.03])
    assert physics.model.geom_size[2, 1] == pytest.approx(0.45)
    assert physics.model.dof_damping[1] == pytest.approx(0.10)
    assert physics.model.dof_damping[0] == pytest.approx(0.10)  # untouched


def test_apply_physics_param_scales_scales_gear_and_gravity() -> None:
    physics = _make_physics()
    physics.model.actuator_gear = np.asarray([[10.0, 0, 0, 0, 0, 0]])
    physics.model.opt = type("Opt", (), {"gravity": np.asarray([0.0, 0.0, -9.81])})()
    apply_physics_param_scales(physics, {"actuator_gear_0": 2.0, "gravity": 0.5})
    assert physics.model.actuator_gear[0, 0] == pytest.approx(20.0)
    np.testing.assert_allclose(physics.model.opt.gravity, [0.0, 0.0, -4.905])
    PixelEnvConfig(physics_param_scales={"actuator_gear_0": 2.0, "gravity": 0.5})


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
    PixelEnvConfig(physics_param_scales={"pole_length": 1.5})
    with pytest.raises(ValueError, match="only supported for cartpole"):
        PixelEnvConfig(env_name="acrobot-swingup", physics_param_scales={"pole_length": 1.5})


def test_pole_length_scale_recompiles_inertia_and_com() -> None:
    """``pole_length`` must change the pole's dynamics, not just its drawing."""

    pytest.importorskip("dm_control")
    from envs.dm_control_pixels import DmControlPixelEnv

    nominal = DmControlPixelEnv(PixelEnvConfig())
    longer = DmControlPixelEnv(PixelEnvConfig(physics_param_scales={"pole_length": 1.5}))
    try:
        nominal_params = nominal.physics_params()
        longer_params = longer.physics_params()
        # Pole is body 2: centre of mass at half length, same declared mass.
        assert nominal_params["body_ipos_2_2"] == pytest.approx(0.5)
        assert longer_params["body_ipos_2_2"] == pytest.approx(0.75)
        assert longer_params["body_mass_2"] == pytest.approx(nominal_params["body_mass_2"])
        assert longer_params["body_inertia_2_0"] > 2.0 * nominal_params["body_inertia_2_0"]
    finally:
        nominal.close()
        longer.close()
