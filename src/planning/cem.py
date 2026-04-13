"""Cross-Entropy Method planner over an offline-trained visual world model.

The planner only needs three things from ``VisualWorldModel``:
- ``imagine_rollout_features(h, z, actions)`` to roll the prior forward in
  latent space for a batch of action sequences sharing the same start state;
- ``reward_head(features)`` to score those rollouts (must be enabled);
- ``initial_posterior(obs)`` / ``posterior_step(h, z, action, obs)`` to drive
  the online MPC loop in ``plan_episode``.

CEM samples ``n_samples`` action sequences from a per-timestep diagonal
Gaussian, scores each by summed (γ-discounted) imagined reward, refits the
distribution to the top ``elite_frac`` and iterates ``n_iters`` times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch

from envs.dm_control_pixels import DmControlPixelEnv
from models import VisualWorldModel


@dataclass(frozen=True)
class CEMPlanResult:
    """Output of one ``CEMPlanner.plan_action_sequence`` call."""

    action_sequence: torch.Tensor
    """Mean action sequence after the final CEM iteration, shaped ``[H, A]``."""

    elite_score_mean: float
    """Mean discounted return across elite samples in the final iteration."""

    elite_score_std: float
    """Std of discounted return across elite samples in the final iteration."""


class CEMPlanner:
    """CEM planner with a Gaussian per-timestep proposal and elite refits."""

    def __init__(
        self,
        model: VisualWorldModel,
        horizon: int,
        action_low: np.ndarray | torch.Tensor,
        action_high: np.ndarray | torch.Tensor,
        n_samples: int = 512,
        n_iters: int = 3,
        elite_frac: float = 0.125,
        discount: float = 0.99,
        noise_std_floor: float = 1e-3,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        if model.reward_head is None:
            raise ValueError(
                "CEMPlanner requires a model trained with --reward-head; "
                "the planner scores rollouts with reward_head(features).",
            )
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if n_iters < 1:
            raise ValueError("n_iters must be >= 1")
        if not 0.0 < elite_frac <= 1.0:
            raise ValueError("elite_frac must satisfy 0 < elite_frac <= 1")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must satisfy 0 <= discount <= 1")
        if noise_std_floor < 0.0:
            raise ValueError("noise_std_floor must be >= 0")

        self.model = model
        self.horizon = horizon
        self.action_dim = model.action_dim
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.elite_frac = elite_frac
        self.discount = discount
        self.noise_std_floor = noise_std_floor
        self.device = torch.device(device)

        action_low_t = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        action_high_t = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        if action_low_t.shape != (self.action_dim,) or action_high_t.shape != (self.action_dim,):
            raise ValueError(
                f"action_low/action_high must have shape [{self.action_dim}]; "
                f"got {tuple(action_low_t.shape)} / {tuple(action_high_t.shape)}",
            )
        if torch.any(action_high_t <= action_low_t):
            raise ValueError("action_high must exceed action_low elementwise")
        self.action_low = action_low_t
        self.action_high = action_high_t

        # Discount weights applied per imagined timestep (k = 0..H-1).
        steps = torch.arange(horizon, device=self.device, dtype=torch.float32)
        self._discount_weights = discount**steps  # [H]

        self._n_elite = max(int(round(n_samples * elite_frac)), 1)
        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(int(seed))

    @torch.no_grad()
    def plan_action_sequence(
        self,
        deter: torch.Tensor,
        stoch: torch.Tensor,
    ) -> CEMPlanResult:
        """Run CEM and return the final mean action sequence + elite score stats.

        ``deter``/``stoch`` are single-state tensors shaped ``[hidden_size]``
        and ``[latent_size]``; they are expanded to ``[n_samples, ...]`` for
        the batched rollout.
        """

        if deter.ndim != 1 or deter.shape[0] != self.model.hidden_size:
            raise ValueError(
                f"deter must have shape [hidden_size={self.model.hidden_size}]",
            )
        if stoch.ndim != 1 or stoch.shape[0] != self.model.latent_size:
            raise ValueError(
                f"stoch must have shape [latent_size={self.model.latent_size}]",
            )

        deter_batch = deter.unsqueeze(0).expand(self.n_samples, -1).contiguous()
        stoch_batch = stoch.unsqueeze(0).expand(self.n_samples, -1).contiguous()

        # Initial proposal: mid-range mean, half-range std → uniform-like.
        mean = ((self.action_low + self.action_high) / 2.0).expand(
            self.horizon, -1
        ).contiguous()
        std = ((self.action_high - self.action_low) / 2.0).expand(
            self.horizon, -1
        ).contiguous()

        was_training = self.model.training
        self.model.eval()

        elite_scores = torch.zeros(self._n_elite, device=self.device)
        try:
            for _ in range(self.n_iters):
                noise = self._sample_normal(
                    (self.n_samples, self.horizon, self.action_dim)
                )
                samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise
                samples = torch.maximum(
                    torch.minimum(samples, self.action_high),
                    self.action_low,
                )
                # Latent rollout: features [N, H, hidden+latent].
                features = self.model.imagine_rollout_features(
                    deter_batch, stoch_batch, samples
                )
                # reward_head returns [..., 1]; squeeze the trailing dim.
                rewards = self.model.reward_head(features).squeeze(-1)  # [N, H]
                scores = (rewards * self._discount_weights).sum(dim=1)  # [N]

                elite_indices = torch.topk(scores, self._n_elite).indices
                elite_samples = samples[elite_indices]  # [n_elite, H, A]
                elite_scores = scores[elite_indices]
                mean = elite_samples.mean(dim=0)
                std = elite_samples.std(dim=0).clamp_min(self.noise_std_floor)
        finally:
            if was_training:
                self.model.train()

        return CEMPlanResult(
            action_sequence=mean.detach(),
            elite_score_mean=float(elite_scores.mean().item()),
            elite_score_std=float(elite_scores.std(unbiased=False).item()),
        )

    @torch.no_grad()
    def plan(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        """Convenience: return only the first action of the planned sequence."""

        return self.plan_action_sequence(deter, stoch).action_sequence[0]

    def _sample_normal(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Sample standard normal, honoring the planner's seeded generator."""

        if self._generator is None:
            return torch.randn(*shape, device=self.device)
        return torch.randn(*shape, generator=self._generator, device=self.device)


@dataclass
class EpisodeRollout:
    """One MPC-controlled episode's transcript."""

    rewards: list[float]
    actions: list[np.ndarray]
    images: list[np.ndarray]
    done: bool
    elite_score_means: list[float]
    elite_score_stds: list[float]

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

    @property
    def length(self) -> int:
        return len(self.rewards)


@torch.no_grad()
def plan_episode(
    env: DmControlPixelEnv,
    planner: CEMPlanner,
    max_steps: int,
    image_to_tensor: Callable[[np.ndarray], torch.Tensor] | None = None,
    record_images: bool = False,
) -> EpisodeRollout:
    """Run one MPC-controlled episode in the given env using ``planner``.

    The hidden state ``(h, z)`` is initialized from the env's initial
    observation via ``model.initial_posterior``. After each env step the
    model's posterior is advanced with the executed action and the new
    observation. Replans every step (true MPC).

    ``image_to_tensor`` converts the env's ``uint8`` frame to the float tensor
    the model expects. Defaults to ``image / 255 → CHW float32`` on the
    planner's device.
    """

    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    image_to_tensor = image_to_tensor or _default_image_to_tensor(planner.device)

    initial_image = env.reset()
    obs = image_to_tensor(initial_image)
    deter, stoch = planner.model.initial_posterior(obs)

    rewards: list[float] = []
    actions: list[np.ndarray] = []
    images: list[np.ndarray] = [initial_image] if record_images else []
    elite_means: list[float] = []
    elite_stds: list[float] = []

    done = False
    for _ in range(max_steps):
        if done:
            break
        plan_result = planner.plan_action_sequence(deter[0], stoch[0])
        elite_means.append(plan_result.elite_score_mean)
        elite_stds.append(plan_result.elite_score_std)
        action_tensor = plan_result.action_sequence[0]
        action = action_tensor.detach().cpu().numpy().astype(np.float32)

        step = env.step(action)
        rewards.append(step.reward)
        actions.append(action)
        if record_images:
            images.append(step.image)
        done = step.done

        if not done:
            obs_next = image_to_tensor(step.image)
            deter, stoch = planner.model.posterior_step(
                deter, stoch, action_tensor.unsqueeze(0), obs_next
            )

    return EpisodeRollout(
        rewards=rewards,
        actions=actions,
        images=images,
        done=done,
        elite_score_means=elite_means,
        elite_score_stds=elite_stds,
    )


def _default_image_to_tensor(device: torch.device) -> Callable[[np.ndarray], torch.Tensor]:
    """Default ``uint8 [H, W, 3]`` → float32 tensor ``[1, 3, H, W]`` on device."""

    def convert(image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got dtype {image.dtype}")
        as_float = image.astype(np.float32) / 255.0
        # HWC → CHW, add batch dim.
        return torch.from_numpy(as_float).permute(2, 0, 1).unsqueeze(0).to(device)

    return convert


def aggregate_returns(rollouts: Iterable[EpisodeRollout]) -> dict[str, float]:
    """Reduce a sequence of ``EpisodeRollout`` to a small metrics dict."""

    rollouts_list = list(rollouts)
    if not rollouts_list:
        return {"episodes": 0.0, "return_mean": 0.0, "return_std": 0.0, "length_mean": 0.0}
    returns = np.array([r.total_reward for r in rollouts_list], dtype=np.float64)
    lengths = np.array([r.length for r in rollouts_list], dtype=np.float64)
    return {
        "episodes": float(len(rollouts_list)),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "length_mean": float(lengths.mean()),
    }
