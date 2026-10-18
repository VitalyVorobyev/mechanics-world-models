"""Pixel observation wrapper for DeepMind Control Suite environments."""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from dm_control import suite
from numpy.typing import NDArray


_BODY_MASS_RE = re.compile(r"^body_mass_(\d+)$")
_DOF_DAMPING_RE = re.compile(r"^dof_damping_(\d+)$")
_GEOM_SIZE_RE = re.compile(r"^geom_size_(\d+)_(\d+)$")
# Applied when the MJCF is compiled (not post-hoc), so MuJoCo recomputes the
# pole's inertia and centre of mass. Cartpole only.
POLE_LENGTH_KEY = "pole_length"
_CARTPOLE_POLE_FROMTO = 'fromto="0 0 0 0 0 1"'


@dataclass(frozen=True)
class PixelEnvConfig:
    """Configuration for a dm_control pixel environment.

    ``physics_param_scales`` is a flat dict of positive multiplicative scales
    used for OOD physics shifts. ``None`` keeps the nominal physics. Keys:

    * ``pole_length`` (cartpole only): scales the pole capsule in the MJCF
      before compilation, so length, inertia and centre of mass change
      consistently. This is the only correct way to change the length.
    * ``body_mass_<i>``: scales the body's mass *and* its inertia tensor
      (uniform density change); applied after loading.
    * ``dof_damping_<i>``: scales joint damping; applied after loading.
    * ``geom_size_<i>_<j>``: scales a geom size entry after compilation. This
      only changes collision and rendering — inertia and centre of mass are
      fixed at compile time — so it is **not** a length change. (Before the
      October 2026 fix, ``geom_size_2_1`` was used as "pole length", but geom
      2 is the static rail ``rail2``; the pole is geom 4.)
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
                if key == POLE_LENGTH_KEY:
                    if not self.env_name.startswith("cartpole-"):
                        raise ValueError("pole_length scaling is only supported for cartpole")
                    continue
                if not (
                    _BODY_MASS_RE.match(key)
                    or _DOF_DAMPING_RE.match(key)
                    or _GEOM_SIZE_RE.match(key)
                ):
                    raise ValueError(
                        f"unknown physics_param_scales key {key!r}; expected "
                        "pole_length, body_mass_<i>, dof_damping_<i>, or geom_size_<i>_<j>",
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
        scales = dict(config.physics_param_scales or {})
        pole_length_scale = scales.pop(POLE_LENGTH_KEY, None)
        if pole_length_scale is not None:
            self._env = _load_cartpole_with_pole_length(
                task_name, seed=config.seed, length_scale=pole_length_scale,
            )
        else:
            self._env = suite.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs={"random": config.seed},
            )
        if scales:
            apply_physics_param_scales(self._env.physics, scales)
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

    def reset_without_render(self) -> dict[str, NDArray[np.float64]]:
        """Reset and return the initial physics state, no image (simulator-only diagnostics)."""

        self._time_step = self._env.reset()
        return self.physics_state()

    def episode_done(self) -> bool:
        """Whether the last step ended the episode."""

        return bool(self._time_step is not None and self._time_step.last())

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

    def step_without_render(self, action: NDArray[np.floating[Any]]) -> dict[str, NDArray[np.float64]]:
        """Apply one action for ``action_repeat`` physics steps; return the state, no image.

        For simulator-only diagnostics (e.g. intervention-effect checks)
        where rendering would dominate the cost.
        """

        for _ in range(self.config.action_repeat):
            self._time_step = self._env.step(action)
            if self._time_step.last():
                break
        return self.physics_state()

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
        body_inertia = np.asarray(physics.model.body_inertia, dtype=np.float64)
        body_ipos = np.asarray(physics.model.body_ipos, dtype=np.float64)
        for i in range(body_inertia.shape[0]):
            for j in range(body_inertia.shape[1]):
                params[f"body_inertia_{i}_{j}"] = float(body_inertia[i, j])
            # Centre-of-mass offset along the body z axis (pole: half length).
            params[f"body_ipos_{i}_2"] = float(body_ipos[i, 2])
        dof_damping = np.asarray(physics.model.dof_damping, dtype=np.float64)
        for i, damping in enumerate(dof_damping):
            params[f"dof_damping_{i}"] = float(damping)
        geom_size = np.asarray(physics.model.geom_size, dtype=np.float64)
        for i in range(geom_size.shape[0]):
            for j in range(geom_size.shape[1]):
                params[f"geom_size_{i}_{j}"] = float(geom_size[i, j])
        return params

    def frame_dt(self) -> float:
        """Simulated seconds between consecutive rendered frames.

        Equals the suite's control timestep times ``action_repeat``; this is
        the ``dt`` a continuous-time world model must integrate over.
        """

        return float(self._env.control_timestep()) * self.config.action_repeat

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


def simulator_versions() -> dict[str, str]:
    """Installed versions of the packages that determine physics and pixels."""

    versions: dict[str, str] = {}
    for package in ("mujoco", "dm-control"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _load_cartpole_with_pole_length(task_name: str, *, seed: int, length_scale: float) -> Any:
    """Build a cartpole environment whose pole capsule is ``length_scale`` x nominal.

    Mirrors ``dm_control.suite.cartpole.<task>()`` but edits the MJCF first, so
    MuJoCo compiles the pole's inertia and centre of mass for the new length
    (the geom keeps its declared mass).
    """

    from dm_control.rl import control
    from dm_control.suite import cartpole

    if task_name not in ("balance", "balance_sparse", "swingup", "swingup_sparse"):
        raise ValueError(f"pole_length scaling does not support cartpole task {task_name!r}")
    xml, assets = cartpole.get_model_and_assets()
    xml_text = xml.decode() if isinstance(xml, bytes) else xml
    if xml_text.count(_CARTPOLE_POLE_FROMTO) != 1:
        raise RuntimeError("unexpected cartpole MJCF: cannot locate the pole fromto")
    xml_text = xml_text.replace(
        _CARTPOLE_POLE_FROMTO, f'fromto="0 0 0 0 0 {length_scale:.6g}"',
    )
    physics = cartpole.Physics.from_xml_string(xml_text, assets)
    task = cartpole.Balance(
        swing_up=task_name.startswith("swingup"),
        sparse=task_name.endswith("sparse"),
        random=seed,
    )
    return control.Environment(physics, task, time_limit=cartpole._DEFAULT_TIME_LIMIT)


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
            # Same geometry, scaled density: the inertia tensor scales with mass.
            for axis in range(3):
                _scale_array_entry(physics.model.body_inertia, (index, axis), scale, key)
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
