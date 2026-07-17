# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Small differentiable reference tracer for GLINT 2D Gaussians.

This backend is intentionally quadratic in rays and Gaussians. It provides an
executable specification and gradient tests for the transport pipeline; use the
OptiX backend for real scenes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .checkpoint import GlintGaussianSet
from .geometry import quaternion_to_rotation_matrix
from .spherical_harmonics import evaluate_spherical_harmonics
from .transport import RayBundle, TraceResult


class TorchSurfelTracer(nn.Module):
    """All-pairs 2D Gaussian tracer used for correctness tests and tiny scenes."""

    def __init__(
        self,
        *,
        max_ray_gaussian_pairs: int = 2_000_000,
        alpha_min: float = 1.0 / 255.0,
        max_alpha: float = 0.99,
        intersection_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.max_ray_gaussian_pairs = max_ray_gaussian_pairs
        self.alpha_min = alpha_min
        self.max_alpha = max_alpha
        self.intersection_epsilon = intersection_epsilon

    def trace(
        self,
        gaussians: GlintGaussianSet,
        rays: RayBundle,
        *,
        background: Tensor,
    ) -> TraceResult:
        height, width, _ = rays.origins.shape
        origins = rays.origins.reshape(-1, 3)
        directions = rays.directions.reshape(-1, 3)
        means = gaussians.get_xyz
        n_rays, n_gaussians = origins.shape[0], means.shape[0]
        pair_count = n_rays * n_gaussians
        if pair_count > self.max_ray_gaussian_pairs:
            raise RuntimeError(
                "TorchSurfelTracer is an O(rays * Gaussians) reference backend: "
                f"requested {pair_count:,} pairs, limit is "
                f"{self.max_ray_gaussian_pairs:,}. Install/use the OptiX backend "
                "for full GLINT scenes."
            )

        rotations = quaternion_to_rotation_matrix(gaussians.get_rotation)
        tangent_u = rotations[:, :, 0]
        tangent_v = rotations[:, :, 1]
        surfel_normals = rotations[:, :, 2]
        scales = gaussians.get_scaling[..., :2].clamp_min(1e-8)

        ray_o = origins[:, None, :]
        ray_d = directions[:, None, :]
        normals = surfel_normals[None, :, :]
        denominator = (ray_d * normals).sum(dim=-1)
        nonparallel = denominator.abs() > self.intersection_epsilon
        safe_denominator = torch.where(
            nonparallel, denominator, torch.ones_like(denominator)
        )
        distance = ((means[None, :, :] - ray_o) * normals).sum(
            dim=-1
        ) / safe_denominator
        valid = nonparallel & (distance > self.intersection_epsilon)

        points = ray_o + distance[..., None] * ray_d
        offsets = points - means[None, :, :]
        u = (offsets * tangent_u[None, :, :]).sum(dim=-1) / scales[None, :, 0]
        v = (offsets * tangent_v[None, :, :]).sum(dim=-1) / scales[None, :, 1]
        alpha = gaussians.get_opacity.reshape(1, -1) * torch.exp(
            -0.5 * (u.square() + v.square())
        )
        alpha = alpha.clamp(max=self.max_alpha)
        alpha = torch.where(valid & (alpha >= self.alpha_min), alpha, 0.0)

        sort_distance = torch.where(
            valid, distance, torch.full_like(distance, float("inf"))
        )
        sorted_distance, order = sort_distance.sort(dim=-1)
        sorted_alpha = alpha.gather(-1, order)
        transmittance = torch.cumprod(
            torch.cat(
                (
                    torch.ones_like(sorted_alpha[:, :1]),
                    1.0 - sorted_alpha[:, :-1],
                ),
                dim=-1,
            ),
            dim=-1,
        )
        weights = transmittance * sorted_alpha

        expanded_directions = directions[:, None, :].expand(-1, n_gaussians, -1)
        colors = torch.clamp_min(
            evaluate_spherical_harmonics(
                int(gaussians.active_sh_degree.item()),
                expanded_directions,
                gaussians.get_features,
            )
            + 0.5,
            0.0,
        )
        colors = colors.gather(-2, order[..., None].expand(-1, -1, 3))
        rgb = (weights[..., None] * colors).sum(dim=-2)
        accumulated_alpha = weights.sum(dim=-1, keepdim=True)
        background = background.to(device=rgb.device, dtype=rgb.dtype).flatten()[:3]
        rgb = rgb + (1.0 - accumulated_alpha) * background

        finite_distance = torch.where(
            torch.isfinite(sorted_distance),
            sorted_distance,
            torch.zeros_like(sorted_distance),
        )
        depth = (weights * finite_distance).sum(dim=-1, keepdim=True)
        oriented_normals = torch.where(
            (expanded_directions * surfel_normals[None, :, :]).sum(dim=-1, keepdim=True)
            > 0,
            -surfel_normals[None, :, :],
            surfel_normals[None, :, :],
        )
        oriented_normals = oriented_normals.gather(
            -2, order[..., None].expand(-1, -1, 3)
        )
        normal = (weights[..., None] * oriented_normals).sum(dim=-2)

        specular = None
        transparency = None
        if gaussians.has_specular:
            sorted_specular = (
                gaussians.get_specular.reshape(1, -1)
                .expand(n_rays, -1)
                .gather(-1, order)
            )
            specular = (weights * sorted_specular).sum(dim=-1, keepdim=True)
        if gaussians.has_transparency:
            sorted_transparency = (
                gaussians.get_transmission_coeff.reshape(1, -1)
                .expand(n_rays, -1)
                .gather(-1, order)
            )
            transparency = (weights * sorted_transparency).sum(dim=-1, keepdim=True)

        def image(value: Tensor) -> Tensor:
            return value.reshape(height, width, value.shape[-1])

        return TraceResult(
            rgb=image(rgb),
            depth=image(depth),
            alpha=image(accumulated_alpha),
            normal=image(normal),
            specular=image(specular) if specular is not None else None,
            transparency=(image(transparency) if transparency is not None else None),
            backend_data={
                "weights": weights,
                "gaussian_order": order,
                "pair_count": pair_count,
            },
        )
