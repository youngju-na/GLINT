# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Adapter from GLINT Gaussian sets to the differentiable OptiX surfel tracer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .checkpoint import GlintGaussianSet
from .geometry import quaternion_to_rotation_matrix
from .transport import RayBundle, TraceResult


@dataclass
class _TracerState:
    tracer: Any
    version: tuple[int, ...]


def make_surfel_triangles(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    *,
    sigma_extent: float = 3.0,
) -> tuple[Tensor, Tensor]:
    """Convert 2D Gaussians into the two-triangle disks used by GLINT's BVH."""

    rotations = quaternion_to_rotation_matrix(quats)
    tangent_u = rotations[:, :, 0] * scales[:, :1]
    tangent_v = rotations[:, :, 1] * scales[:, 1:2]
    corners = means.new_tensor(
        [[-1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, -1.0]]
    ) * float(sigma_extent)
    vertices = (
        means[:, None, :]
        + corners[None, :, :1] * tangent_u[:, None, :]
        + corners[None, :, 1:] * tangent_v[:, None, :]
    )
    indices = torch.arange(
        vertices.shape[0] * 4, dtype=torch.int32, device=means.device
    ).reshape(-1, 4)
    triangles = torch.stack((indices[:, :3], indices[:, 1:]), dim=1)
    return vertices.reshape(-1, 3).contiguous(), triangles.reshape(-1, 3).contiguous()


class OptixSurfelTracer(nn.Module):
    """Full-scene arbitrary-ray backend using GLINT's OptiX extension."""

    def __init__(self, *, sigma_extent: float = 3.0) -> None:
        super().__init__()
        try:
            from diff_surfel_tracing import SurfelTracer, SurfelTracingSettings
        except ImportError as exc:
            raise ImportError(
                "OptixSurfelTracer requires the optional diff_surfel_tracing "
                "extension. Build the recursive dependency described in "
                "README.md."
            ) from exc
        self._surfel_tracer_type = SurfelTracer
        self._settings_type = SurfelTracingSettings
        self.sigma_extent = sigma_extent
        self._states: dict[int, _TracerState] = {}

    @staticmethod
    def _version(gaussians: GlintGaussianSet) -> tuple[int, ...]:
        return (
            gaussians._xyz.data_ptr(),
            gaussians._xyz._version,
            gaussians._scaling.data_ptr(),
            gaussians._scaling._version,
            gaussians._rotation.data_ptr(),
            gaussians._rotation._version,
        )

    def _prepare_tracer(
        self, gaussians: GlintGaussianSet, vertices: Tensor, triangles: Tensor
    ) -> Any:
        key = id(gaussians)
        version = self._version(gaussians)
        state = self._states.get(key)
        if state is None:
            state = _TracerState(tracer=self._surfel_tracer_type(), version=(-1,) * 6)
            self._states[key] = state
        state.tracer.train(self.training)
        if state.version != version:
            state.tracer.build_acceleration_structure(
                vertices.detach().clone(), triangles.detach().clone(), rebuild=True
            )
            state.version = version
        return state.tracer

    def trace(
        self,
        gaussians: GlintGaussianSet,
        rays: RayBundle,
        *,
        background: Tensor,
    ) -> TraceResult:
        means = gaussians.get_xyz.contiguous()
        scales = gaussians.get_scaling[..., :2].contiguous()
        quats = gaussians.get_rotation.contiguous()
        vertices, triangles = make_surfel_triangles(
            means, quats, scales, sigma_extent=self.sigma_extent
        )
        tracer = self._prepare_tracer(gaussians, vertices, triangles)

        device, dtype = means.device, means.dtype
        height, width, _ = rays.origins.shape
        settings = self._settings_type(
            image_height=height,
            image_width=width,
            tanfovx=0.0,
            tanfovy=0.0,
            bg=background.to(device=device, dtype=dtype).flatten()[:3].contiguous(),
            scale_modifier=1.0,
            viewmatrix=torch.eye(4, device=device, dtype=dtype),
            projmatrix=torch.eye(4, device=device, dtype=dtype),
            sh_degree=int(gaussians.active_sh_degree.item()),
            campos=torch.zeros(3, device=device, dtype=dtype),
            prefiltered=False,
            debug=False,
            max_trace_depth=0,
            specular_threshold=0.0,
        )
        grads3d = torch.zeros_like(means, requires_grad=True)
        others = None
        if gaussians.has_specular:
            transparency = (
                gaussians.get_transmission_coeff
                if gaussians.has_transparency
                else torch.zeros_like(gaussians.get_specular)
            )
            others = torch.cat(
                (gaussians.get_specular, transparency), dim=-1
            ).contiguous()

        rgb, depth, alpha, normal, distance, aux, mid, weights = tracer(
            rays.origins.contiguous(),
            rays.directions.contiguous(),
            vertices,
            means3D=means,
            grads3D=grads3d,
            shs=gaussians.get_features.contiguous(),
            colors_precomp=None,
            others_precomp=others,
            opacities=gaussians.get_opacity.contiguous(),
            scales=scales,
            rotations=quats,
            cov3D_precomp=None,
            tracer_settings=settings,
            start_from_first=False,
        )
        return TraceResult(
            rgb=rgb,
            depth=depth,
            alpha=alpha,
            normal=normal,
            specular=(aux[..., :1] if gaussians.has_specular else None),
            transparency=(aux[..., 1:2] if gaussians.has_transparency else None),
            weight_accumulate=weights,
            backend_data={
                "distance": distance,
                "mid": mid,
                "viewspace_points": grads3d,
                "vertices": vertices,
            },
        )
