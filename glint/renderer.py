# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Compatibility layer between GLINT's EasyVolCap renderer and gsplat.

The public entry point :func:`render_glint_camera` intentionally uses duck typing:
it accepts GLINT's existing camera, GaussianModel, and pipeline objects without
importing EasyVolCap. This keeps the original GLINT checkout independent while we
move one rendering path at a time into gsplat.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat import spherical_harmonics
from gsplat.rendering import rasterization_2dgs


class AttrDict(dict):
    """Small dict with the attribute access used by EasyVolCap's ``dotdict``."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _single_camera_matrix(matrix: Tensor, name: str, size: int) -> Tensor:
    if matrix.shape == (size, size):
        return matrix.unsqueeze(0)
    if matrix.shape == (1, size, size):
        return matrix
    raise ValueError(
        f"GLINT compatibility currently renders one camera at a time; "
        f"{name} must be [{size}, {size}] or [1, {size}, {size}], got "
        f"{tuple(matrix.shape)}"
    )


def _prepare_scales(scales: Tensor, scaling_modifier: float) -> Tensor:
    if scales.ndim != 2 or scales.shape[-1] not in (2, 3):
        raise ValueError(f"scales must be [N, 2] or [N, 3], got {tuple(scales.shape)}")

    # GLINT stores only the two tangent-plane scales. gsplat represents a 2D
    # Gaussian with a 3-vector and expects the normal-axis scale to be one.
    tangent_scales = scales[..., :2] * scaling_modifier
    normal_scale = torch.ones_like(tangent_scales[..., :1])
    return torch.cat((tangent_scales, normal_scale), dim=-1)


def _prepare_background(
    background: Optional[Tensor], channels: int, reference: Tensor
) -> Optional[Tensor]:
    if background is None:
        return None
    background = background.to(device=reference.device, dtype=reference.dtype).flatten()
    if background.numel() > channels:
        background = background[:channels]
    elif background.numel() < channels:
        # GLINT's material channels have a black/zero background. Some existing
        # configs provide only the three RGB values when material channels render.
        background = F.pad(background, (0, channels - background.numel()))
    return background.unsqueeze(0)


def _gradient_isolated_feature(value: Tensor, detached_input_value: Tensor) -> Tensor:
    """Keep ``value`` in the forward pass but cancel geometry-weight gradients.

    ``detached_input_value`` must be the rasterized copy of the same feature
    whose per-Gaussian input was detached.  Both channels therefore have the
    same forward value and the same compositing-weight derivative, while only
    ``value`` retains a derivative with respect to the material attribute.
    """

    return value - detached_input_value + detached_input_value.detach()


def _glint_depth_to_normal(depth: Tensor, camtoworld: Tensor, K: Tensor) -> Tensor:
    """Match GLINT's pixel-center convention when converting z-depth to normals."""

    if depth.ndim != 4 or depth.shape[0] != 1 or depth.shape[-1] != 1:
        raise ValueError(f"depth must be [1, H, W, 1], got {tuple(depth.shape)}")

    _, height, width, _ = depth.shape
    dtype, device = depth.dtype, depth.device
    x, y = torch.meshgrid(
        torch.arange(width, dtype=dtype, device=device),
        torch.arange(height, dtype=dtype, device=device),
        indexing="xy",
    )
    camera_dirs = torch.stack(
        (
            (x - K[0, 0, 2]) / K[0, 0, 0],
            (y - K[0, 1, 2]) / K[0, 1, 1],
            torch.ones_like(x),
        ),
        dim=-1,
    )
    world_dirs = torch.einsum("ij,hwj->hwi", camtoworld[0, :3, :3], camera_dirs)
    points = camtoworld[0, :3, 3][None, None, :] + depth[0] * world_dirs
    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    normals = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    return F.pad(normals, (0, 0, 1, 1, 1, 1), value=0.0).unsqueeze(0)


def _make_viewspace_proxy(
    means: Tensor,
    densify: Tensor,
    selected: Optional[Tensor],
    width: int,
    height: int,
) -> Tensor:
    """Expose gsplat's 2D densification gradient in GLINT's legacy [N, 3] slot."""

    proxy = torch.zeros_like(means, requires_grad=torch.is_grad_enabled())
    if not densify.requires_grad:
        return proxy
    densify.retain_grad()

    def capture_gradient(gradient: Tensor) -> Tensor:
        gradient_2d = gradient[0] if gradient.ndim == 3 else gradient
        # gsplat exposes normalized screen-space gradients and its native
        # densification strategy converts them back to pixel space. GLINT's
        # means2D proxy already receives pixel-space gradients.
        gradient_2d = gradient_2d.clone()
        gradient_2d[..., 0] *= width / 2.0
        gradient_2d[..., 1] *= height / 2.0
        proxy_gradient = torch.zeros_like(proxy)
        if selected is None:
            proxy_gradient[..., :2] = gradient_2d
        else:
            proxy_gradient[selected, :2] = gradient_2d
        proxy.grad = proxy_gradient.detach()
        return gradient

    densify.register_hook(capture_gradient)
    return proxy


def rasterize_glint_2dgs(
    *,
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    features: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    background: Optional[Tensor] = None,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    scaling_modifier: float = 1.0,
    depth_ratio: float = 0.0,
    specular_channels: Optional[int] = None,
    trans_mask: Optional[Tensor] = None,
) -> AttrDict:
    """Render GLINT's interaction Gaussians with gsplat's native 2DGS backend.

    Inputs use GLINT's activated parameter convention: two tangent-plane scales,
    opacity in ``[0, 1]``, and precomputed RGB/material features. The returned
    fields mirror ``easyvolcap.utils.gaussian2d_utils.render`` closely enough for
    GLINT's existing output post-processing and losses.

    Exact per-Gaussian accumulated compositing weights are not yet exposed by
    gsplat, so the legacy ``weight_accumulate`` field is deliberately absent.
    GLINT already treats that field as optional during densification.
    """

    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"means must be [N, 3], got {tuple(means.shape)}")
    n_gaussians = means.shape[0]
    if quats.shape != (n_gaussians, 4):
        raise ValueError(f"quats must be [N, 4], got {tuple(quats.shape)}")
    if features.ndim != 2 or features.shape[0] != n_gaussians:
        raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")
    if not 0.0 <= depth_ratio <= 1.0:
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")

    # Render a detached duplicate of interface transparency in the same fused
    # rasterization.  Combining the two output channels below leaves the
    # forward value unchanged while canceling transparency-routing gradients
    # to means, covariance, and opacity.  One scalar feature is substantially
    # cheaper than another rasterization pass.
    if specular_channels is not None:
        expected_channels = 3 + specular_channels + 1
        if features.shape[-1] != expected_channels:
            raise ValueError(
                "GLINT material input expects RGB + specular channels + "
                f"transparency = {expected_channels} channels, got "
                f"{features.shape[-1]}"
            )
        transparency_start = 3 + specular_channels
        transparency_input = features[:, transparency_start : transparency_start + 1]
        features = torch.cat((features, transparency_input.detach()), dim=-1)

    viewmats = _single_camera_matrix(viewmat, "viewmat", 4)
    Ks = _single_camera_matrix(K, "K", 3)
    scales_3d = _prepare_scales(scales, scaling_modifier)
    opacities = opacities.reshape(-1)
    if opacities.shape[0] != n_gaussians:
        raise ValueError(
            f"opacities must contain N values, got {tuple(opacities.shape)}"
        )

    selected = None
    if trans_mask is not None:
        selected = trans_mask.to(device=means.device, dtype=torch.bool).flatten()
        if selected.shape[0] != n_gaussians:
            raise ValueError(
                f"trans_mask must contain N values, got {tuple(selected.shape)}"
            )
        means_render = means[selected]
        quats_render = quats[selected]
        scales_render = scales_3d[selected]
        opacities_render = opacities[selected]
        features_render = features[selected]
    else:
        means_render = means
        quats_render = quats
        scales_render = scales_3d
        opacities_render = opacities
        features_render = features

    backgrounds = _prepare_background(background, features.shape[-1], features)
    (
        render_with_depth,
        render_alpha,
        render_normal,
        _native_surface_normal,
        render_distortion,
        render_median,
        meta,
    ) = rasterization_2dgs(
        means=means_render,
        quats=quats_render,
        scales=scales_render,
        opacities=opacities_render,
        # Per-camera shape avoids a native RGB+ED concatenation ambiguity for
        # post-activation features with more than three channels.
        colors=features_render.unsqueeze(0),
        viewmats=viewmats,
        Ks=Ks,
        width=int(width),
        height=int(height),
        near_plane=float(near_plane),
        far_plane=float(far_plane),
        backgrounds=backgrounds,
        packed=False,
        render_mode="RGB+ED",
        distloss=True,
        depth_mode="expected" if depth_ratio < 0.5 else "median",
    )

    rendered_features = render_with_depth[..., :-1]
    expected_depth = torch.nan_to_num(render_with_depth[..., -1:], 0.0, 0.0)
    median_depth = torch.nan_to_num(render_median, 0.0, 0.0)
    surface_depth = expected_depth * (1.0 - depth_ratio) + median_depth * depth_ratio
    camtoworlds = torch.linalg.inv(viewmats)
    surface_normal = _glint_depth_to_normal(surface_depth, camtoworlds, Ks)
    surface_normal = surface_normal * render_alpha.detach()

    radii_render = meta["radii"][0].amax(dim=-1).to(dtype=means.dtype)
    full_radii = torch.zeros(n_gaussians, dtype=means.dtype, device=means.device)
    visibility = torch.zeros(n_gaussians, dtype=torch.bool, device=means.device)
    if selected is None:
        full_radii = radii_render
        visibility = radii_render > 0
    else:
        full_radii[selected] = radii_render
        visibility[selected] = radii_render > 0

    viewspace_points = _make_viewspace_proxy(
        means, meta["gradient_2dgs"], selected, int(width), int(height)
    )
    rendered_chw = rendered_features[0].permute(2, 0, 1)
    alpha_chw = render_alpha[0].permute(2, 0, 1)
    output = AttrDict(
        render=rendered_chw[:3],
        rend_alpha=alpha_chw,
        rend_normal=render_normal[0].permute(2, 0, 1),
        rend_dist=render_distortion[0].permute(2, 0, 1),
        surf_depth=surface_depth[0].permute(2, 0, 1),
        surf_normal=surface_normal[0].permute(2, 0, 1),
        render_depth_expected=expected_depth[0].permute(2, 0, 1),
        render_depth_median=median_depth[0].permute(2, 0, 1),
        viewspace_points=viewspace_points,
        visibility_filter=visibility,
        radii=full_radii,
        native_meta=meta,
        has_exact_weight_accumulate=False,
    )

    if specular_channels is not None:
        expected_channels = 3 + specular_channels + 2
        if rendered_chw.shape[0] != expected_channels:
            raise ValueError(
                "GLINT material layout expects RGB + specular channels + "
                "transparency + detached transparency = "
                f"{expected_channels} channels, got "
                f"{rendered_chw.shape[0]}"
            )
        output.specular = rendered_chw[3 : 3 + specular_channels]
        transparency = rendered_chw[3 + specular_channels : 3 + specular_channels + 1]
        detached_transparency = rendered_chw[
            3 + specular_channels + 1 : 3 + specular_channels + 2
        ]
        output.transparency = _gradient_isolated_feature(
            transparency,
            detached_transparency,
        )
        # The composited gate alpha*t is correct for transport, whereas
        # material supervision should estimate t conditioned on an interface
        # hit.  Detaching alpha prevents that supervision from increasing
        # coverage as an alternative to changing the material parameter.
        coverage = alpha_chw.detach()
        output.conditional_transparency = torch.where(
            coverage > 1e-6,
            output.transparency / coverage.clamp_min(1e-6),
            torch.zeros_like(output.transparency),
        ).clamp(0.0, 1.0)

    return output


def render_glint_camera(
    viewpoint_camera: Any,
    pc: Any,
    pipe: Any,
    bg_color: Tensor,
    scaling_modifier: float = 1.0,
    override_color: Optional[Tensor] = None,
    trans_mask: Optional[Tensor] = None,
    device: str = "cuda",
) -> AttrDict:
    """Drop-in-oriented adapter for GLINT's existing ``render`` call signature."""

    if bool(getattr(pipe, "compute_cov3D_python", False)):
        raise NotImplementedError(
            "GLINT's compute_cov3D_python path is not part of the first gsplat "
            "integration milestone; pass activated scales and rotations instead."
        )

    means = pc.get_xyz.to(device)
    if override_color is None:
        coeffs = pc.get_features
        camera_center = viewpoint_camera.camera_center.to(
            device=means.device, dtype=means.dtype
        )
        directions = means - camera_center.reshape(1, 3)
        active_degree = getattr(pc, "active_sh_degree", 0)
        if isinstance(active_degree, Tensor):
            active_degree = int(active_degree.item())
        colors = spherical_harmonics(int(active_degree), directions, coeffs)
        colors = torch.clamp_min(colors + 0.5, 0.0)
    else:
        colors = override_color

    render_reflection = bool(getattr(pc, "render_reflection", False))
    specular_channels = None
    if render_reflection:
        specular_channels = int(getattr(pc, "specular_channels", 1))
        colors = torch.cat(
            (
                colors,
                pc.get_specular,
                pc.get_transmission_coeff,
            ),
            dim=-1,
        )

    world_view_transform = viewpoint_camera.world_view_transform.to(
        device=means.device, dtype=means.dtype
    )
    viewmat = world_view_transform.transpose(-1, -2).contiguous()
    if hasattr(viewpoint_camera, "get_k"):
        K = viewpoint_camera.get_k().to(device=means.device, dtype=means.dtype)
    else:
        K = viewpoint_camera.K.to(device=means.device, dtype=means.dtype)

    return rasterize_glint_2dgs(
        means=means,
        quats=pc.get_rotation,
        scales=pc.get_scaling,
        opacities=pc.get_opacity,
        features=colors,
        viewmat=viewmat,
        K=K,
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        background=bg_color,
        near_plane=float(viewpoint_camera.znear),
        far_plane=float(viewpoint_camera.zfar),
        scaling_modifier=scaling_modifier,
        depth_ratio=float(getattr(pipe, "depth_ratio", 0.0)),
        specular_channels=specular_channels,
        trans_mask=trans_mask,
    )
