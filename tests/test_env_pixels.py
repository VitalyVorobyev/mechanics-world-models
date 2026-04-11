from __future__ import annotations

import multiprocessing as mp
import queue

import numpy as np
import pytest


def _is_rendering_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = ("egl", "glfw", "mujoco", "opengl", "osmesa", "render")
    return any(marker in message for marker in markers)


def _run_env_probe(result_queue) -> None:
    from envs import DmControlPixelEnv, PixelEnvConfig

    env = None
    try:
        env = DmControlPixelEnv(
            PixelEnvConfig(seed=0, action_repeat=2, image_size=84),
        )
        image = env.reset()
        action = env.sample_action(np.random.default_rng(0))
        step = env.step(action)
        result_queue.put(
            (
                "ok",
                image.shape,
                image.dtype.str,
                step.image.shape,
                step.image.dtype.str,
                type(step.reward).__name__,
                type(step.discount).__name__,
                type(step.done).__name__,
            ),
        )
    except Exception as exc:
        result_queue.put(("error", repr(exc)))
    finally:
        if env is not None:
            env.close()


def test_dm_control_pixel_env_returns_expected_image_shape_and_dtype() -> None:
    pytest.importorskip("dm_control")

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_run_env_probe, args=(result_queue,))
    process.start()
    process.join(timeout=20)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.skip("dm_control offscreen rendering did not finish within 20s")

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        pytest.fail(f"dm_control render probe exited without a result: {process.exitcode}")

    if result[0] == "error":
        if _is_rendering_failure(Exception(result[1])):
            pytest.skip(f"dm_control offscreen rendering is unavailable: {result[1]}")
        pytest.fail(result[1])

    _, image_shape, image_dtype, step_shape, step_dtype, reward_type, discount_type, done_type = result
    assert image_shape == (84, 84, 3)
    assert np.dtype(image_dtype) == np.uint8
    assert step_shape == (84, 84, 3)
    assert np.dtype(step_dtype) == np.uint8
    assert reward_type == "float"
    assert discount_type == "float"
    assert done_type == "bool"
