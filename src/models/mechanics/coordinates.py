"""Coordinate-topology helpers for generalized coordinates with angle dims.

Cartpole's configuration space is ``R x S^1`` (cart position, pole angle). A
plain ``R^2`` latent cannot represent the circle continuously, and a
non-periodic decoder or potential lets the encoder pick any chart for the
angle. Marking angle dims lets every consumer of ``q`` see the angle only
through ``(cos, sin)`` and lets finite differences / KL means wrap across the
``+-pi`` branch cut.

With no angle dims, every helper reduces to the legacy behaviour exactly, so
checkpoints trained before this module existed load and evaluate unchanged.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def validate_angle_dims(angle_dims: Sequence[int], q_dim: int) -> tuple[int, ...]:
    """Return ``angle_dims`` as a sorted tuple after range/duplicate checks."""

    dims = tuple(sorted(int(d) for d in angle_dims))
    if len(set(dims)) != len(dims):
        raise ValueError(f"angle_dims must not repeat; got {list(angle_dims)}")
    for dim in dims:
        if not 0 <= dim < q_dim:
            raise ValueError(f"angle dim {dim} out of range for q_dim={q_dim}")
    return dims


def coordinate_feature_size(q_dim: int, angle_dims: Sequence[int]) -> int:
    """Width of ``embed_coordinates`` output: each angle dim becomes two features."""

    return q_dim + len(angle_dims)


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """Map angles to ``[-pi, pi)``; gradient is the identity almost everywhere."""

    return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi


def wrap_angle_dims(x: torch.Tensor, angle_dims: Sequence[int]) -> torch.Tensor:
    """Wrap only the ``angle_dims`` entries of the last axis of ``x``.

    Used on *differences* of coordinates (finite-difference velocities, prior
    minus posterior means), where a jump across the branch cut is an artefact.
    """

    if not angle_dims:
        return x
    mask = torch.zeros(x.shape[-1], dtype=torch.bool, device=x.device)
    mask[list(angle_dims)] = True
    return torch.where(mask, wrap_angle(x), x)


def embed_coordinates(
    q: torch.Tensor,
    angle_dims: Sequence[int],
    linear_scale: float | None = None,
) -> torch.Tensor:
    """Return ``[linear dims..., cos(angles)..., sin(angles)...]`` along the last axis.

    ``linear_scale`` applies the dynamics MLPs' ``scale * tanh(x / scale)``
    input bound to the non-angle dims; angle dims are already bounded by
    ``cos``/``sin``. With empty ``angle_dims`` this is just the (optionally
    bounded) ``q``.
    """

    if not angle_dims:
        if linear_scale is None:
            return q
        return linear_scale * torch.tanh(q / linear_scale)
    angle_set = set(angle_dims)
    linear_dims = [i for i in range(q.shape[-1]) if i not in angle_set]
    linear = q[..., linear_dims]
    if linear_scale is not None:
        linear = linear_scale * torch.tanh(linear / linear_scale)
    angles = q[..., list(angle_dims)]
    return torch.cat([linear, torch.cos(angles), torch.sin(angles)], dim=-1)
