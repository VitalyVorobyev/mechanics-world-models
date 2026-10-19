"""Keypoint encoder and geometric renderer for cartpole coordinates.

The October 2026 literature check found that a generic ``Linear -> deconv``
decoder fed a 2-3 number pose cannot render a small moving object: it draws
only the static background (KeyCLD, Daems et al. 2024: "HGN only learns the
background"; Zhong & Leonard 2020: a black-box decoder "fails to reconstruct
rigid bodies in CartPole"; Jaques et al. 2020). They all succeed with a
decoder that *places* the object from the coordinates and an encoder that
*locates* it. This module provides both, specialised to cartpole:

* ``KeypointEncoder``: a fully convolutional net -> two heatmaps (cart, pole
  tip) -> spatial softmax -> image-plane keypoints in ``[-1, 1]``. The
  generalized coordinates are read off geometrically:
  ``x = u_cart`` and ``theta = atan2(u_tip - u_cart, v_cart - v_tip)`` (image
  ``v`` grows downwards, so an upright pole has ``theta = 0``, matching the
  dm_control convention up to the chart's scale/offset).
* ``GeometricRenderer``: computes the cart point and points along the pole
  analytically from ``(x, theta)`` and two learned scalars (cart image height,
  pole image length), splats Gaussian blobs, appends a learned background
  tensor and coordinate channels, and decodes with a small fully
  convolutional net whose limited receptive field forces it to draw near the
  blobs. ``q`` is therefore identifiable up to the renderer's fixed geometry
  and cannot be bypassed by a nuisance channel.

Image coordinates follow ``torch`` conventions: ``u`` horizontal (left = -1),
``v`` vertical (top = -1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _coordinate_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(u, v)`` grids shaped ``[H, W]`` with values in ``[-1, 1]``."""

    v = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    u = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
    return grid_u, grid_v


class KeypointEncoder(nn.Module):
    """Images ``[N, C, 84, 84]`` -> keypoints ``[N, K, 2]`` (u, v) + pooled features.

    Spatial softmax over ``42 x 42`` heatmaps with a learned temperature
    (Finn et al. 2016; KeyCLD). ``num_keypoints = 2`` for cartpole: index 0 is
    the cart, index 1 the pole tip — the assignment is learned, the geometry
    of ``keypoints_to_cartpole_q`` and the renderer pins it down.
    """

    def __init__(
        self,
        num_keypoints: int = 2,
        feature_size: int = 128,
        input_channels: int = 3,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.features = nn.Sequential(
            nn.Conv2d(input_channels + 2, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, hidden_channels, kernel_size=4, stride=2, padding=1),  # 42x42
            nn.ELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ELU(),
        )
        self.heatmap_head = nn.Conv2d(hidden_channels, num_keypoints, kernel_size=1)
        # log of the inverse temperature; sharper maps as training proceeds.
        self.log_inverse_temperature = nn.Parameter(torch.tensor(1.0))
        self.feature_head = nn.Linear(hidden_channels, feature_size)
        self.feature_size = feature_size

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(keypoints [N, K, 2], heatmaps [N, K, h, w], features [N, F])``."""

        if images.ndim != 4:
            raise ValueError("images must be [N, C, H, W]")
        n, _, height, width = images.shape
        grid_u, grid_v = _coordinate_grid(height, width, images.device, images.dtype)
        coords = torch.stack([grid_u, grid_v]).expand(n, -1, -1, -1)
        hidden = self.features(torch.cat([images, coords], dim=1))
        logits = self.heatmap_head(hidden)  # [N, K, h, w]
        h, w = logits.shape[-2:]
        scale = torch.exp(self.log_inverse_temperature)
        heatmaps = torch.softmax((logits * scale).reshape(n, self.num_keypoints, h * w), dim=-1)
        heatmaps = heatmaps.reshape(n, self.num_keypoints, h, w)
        small_u, small_v = _coordinate_grid(h, w, images.device, images.dtype)
        u = (heatmaps * small_u).sum(dim=(-2, -1))
        v = (heatmaps * small_v).sum(dim=(-2, -1))
        keypoints = torch.stack([u, v], dim=-1)  # [N, K, 2]
        features = self.feature_head(hidden.mean(dim=(-2, -1)))
        return keypoints, heatmaps, features


def keypoints_to_cartpole_q(keypoints: torch.Tensor) -> torch.Tensor:
    """``[..., 2, 2]`` (cart, tip) image keypoints -> ``[..., 2]`` ``(x, theta)``."""

    cart = keypoints[..., 0, :]
    tip = keypoints[..., 1, :]
    x = cart[..., 0]
    theta = torch.atan2(tip[..., 0] - cart[..., 0], cart[..., 1] - tip[..., 1])
    return torch.stack([x, theta], dim=-1)


class GeometricRenderer(nn.Module):
    """``(x, theta)`` -> RGB logits ``[N, C, 84, 84]`` via analytic blob placement."""

    def __init__(
        self,
        output_channels: int = 3,
        output_size: int = 84,
        internal_size: int = 42,
        pole_points: int = 8,
        blob_sigma: float = 0.05,
        background_channels: int = 8,
        hidden_channels: int = 32,
        cart_v_init: float = 0.0,
        pole_length_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.output_size = output_size
        self.internal_size = internal_size
        self.blob_sigma = blob_sigma
        self.register_buffer(
            "_pole_fractions", torch.linspace(0.0, 1.0, pole_points), persistent=False,
        )
        self.cart_v = nn.Parameter(torch.tensor(float(cart_v_init)))
        self.log_pole_length = nn.Parameter(torch.tensor(float(pole_length_init)).log())
        self.background = nn.Parameter(
            torch.zeros(background_channels, internal_size, internal_size),
        )
        in_channels = 2 + background_channels + 2  # blobs + background + coords
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(hidden_channels, output_channels, kernel_size=4, stride=2, padding=1),
        )

    @property
    def pole_length(self) -> torch.Tensor:
        return self.log_pole_length.exp()

    def keypoints(self, q: torch.Tensor) -> torch.Tensor:
        """Image keypoints ``[..., 2, 2]`` (cart, tip) implied by ``q = (x, theta)``."""

        x, theta = q[..., 0], q[..., 1]
        cart = torch.stack([x, self.cart_v.expand_as(x)], dim=-1)
        tip = torch.stack(
            [x + self.pole_length * torch.sin(theta), self.cart_v - self.pole_length * torch.cos(theta)],
            dim=-1,
        )
        return torch.stack([cart, tip], dim=-2)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """``q`` ``[N, 2]`` -> logits ``[N, C, output_size, output_size]``."""

        if q.ndim != 2 or q.shape[-1] != 2:
            raise ValueError("q must be [N, 2] (x, theta)")
        n = q.shape[0]
        size = self.internal_size
        grid_u, grid_v = _coordinate_grid(size, size, q.device, q.dtype)
        kp = self.keypoints(q)  # [N, 2, 2]
        cart, tip = kp[:, 0], kp[:, 1]
        fractions = self._pole_fractions.to(q.dtype)
        pole = cart[:, None, :] + fractions[None, :, None] * (tip - cart)[:, None, :]  # [N, P, 2]
        inv_two_var = 1.0 / (2.0 * self.blob_sigma**2)

        def splat(points: torch.Tensor) -> torch.Tensor:
            du = grid_u[None, None] - points[..., 0, None, None]
            dv = grid_v[None, None] - points[..., 1, None, None]
            return torch.exp(-(du.square() + dv.square()) * inv_two_var).amax(dim=1)

        cart_map = splat(cart[:, None, :])  # [N, h, w]
        pole_map = splat(pole)
        coords = torch.stack([grid_u, grid_v]).expand(n, -1, -1, -1)
        features = torch.cat(
            [
                torch.stack([cart_map, pole_map], dim=1),
                self.background.expand(n, -1, -1, -1),
                coords,
            ],
            dim=1,
        )
        logits = self.net(features)
        if logits.shape[-1] != self.output_size:
            logits = F.interpolate(
                logits, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False,
            )
        return logits
