"""Planning backend on the true MuJoCo model (batched native rollouts)."""

from __future__ import annotations

import mujoco
import mujoco.rollout
import numpy as np
from numpy.typing import NDArray

from control.reward import poles_reward

STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS  # [time, qpos, qvel, act]


def get_state(model: mujoco.MjModel, data: mujoco.MjData) -> NDArray[np.float64]:
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


class MujocoBackend:
    """Rolls out ``actions [N, H, A]`` from a full physics state with ``mujoco.rollout``.

    Each action is held for ``action_repeat`` physics steps, as in the
    environment; the reward is the closed-form swing-up reward on every
    physics step (summed per decision), exactly as dm_control scores it, for
    any number of poles.
    """

    def __init__(self, model: mujoco.MjModel, action_repeat: int = 2, nthread: int = 4) -> None:
        self.model = model
        self.action_repeat = action_repeat
        self.datas = [mujoco.MjData(model) for _ in range(nthread)]
        self.roller = mujoco.rollout.Rollout(nthread=nthread)

    def evaluate(
        self, state: NDArray[np.float64], actions: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n, h, _ = actions.shape
        r = self.action_repeat
        control = np.repeat(actions, r, axis=1)                       # [N, H*r, A]
        states, _ = self.roller.rollout(self.model, self.datas, state[None], control)
        nq = self.model.nq
        qpos = states[..., 1:1 + nq]                                  # [N, H*r, nq]
        qvel = states[..., 1 + nq:1 + nq + self.model.nv]
        reward = poles_reward(qpos[..., 0], qpos[..., 1:], qvel[..., 1:], control[..., 0])
        reward = reward.reshape(n, h, r).sum(axis=-1)
        cart_x = qpos[..., 0].reshape(n, h, r)[..., -1]
        return reward, cart_x

    def close(self) -> None:
        self.roller.close() if hasattr(self.roller, "close") else None
