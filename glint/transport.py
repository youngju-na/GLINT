# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""GLINT's transparency-aware Gaussian radiance transport.

This module implements the method-level part of GLINT independently from
EasyVolCap and from a particular arbitrary-ray tracing implementation. The
interface set is rasterized by gsplat; transmission and reflection sets are
queried through the :class:`GlintRayTracer` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from .checkpoint import GlintCheckpoint, GlintGaussianSet
from .renderer import AttrDict, render_glint_camera


@dataclass(frozen=True)
class RayBundle:
    """A dense image of arbitrary world-space rays in ``[H, W, 3]`` layout."""

    origins: Tensor
    directions: Tensor

    def __post_init__(self) -> None:
        if self.origins.shape != self.directions.shape:
            raise ValueError(
                "Ray origins and directions must have the same shape, got "
                f"{tuple(self.origins.shape)} and {tuple(self.directions.shape)}"
            )
        if self.origins.ndim != 3 or self.origins.shape[-1] != 3:
            raise ValueError(
                f"Dense ray bundles must be [H, W, 3], got {tuple(self.origins.shape)}"
            )


@dataclass
class TraceResult:
    """Common result returned by arbitrary-ray Gaussian tracing backends."""

    rgb: Tensor
    depth: Tensor
    alpha: Tensor
    normal: Tensor
    specular: Optional[Tensor] = None
    transparency: Optional[Tensor] = None
    weight_accumulate: Optional[Tensor] = None
    backend_data: Any = None


class GlintRayTracer(Protocol):
    """Backend contract for querying a 2D Gaussian set along arbitrary rays."""

    def trace(
        self,
        gaussians: GlintGaussianSet,
        rays: RayBundle,
        *,
        background: Tensor,
    ) -> TraceResult:
        ...


@dataclass(frozen=True)
class TransportWeights:
    """Energy-conserving per-pixel weights for GLINT's radiance paths."""

    diffuse: Tensor
    reflection: Tensor
    transmission: Tensor
    secondary_reflection: Tensor
    fresnel: Tensor
    energy_sum: Tensor


@dataclass(frozen=True)
class RadianceComposition:
    """Final radiance and already-weighted path contributions."""

    rgb: Tensor
    diffuse: Tensor
    reflection: Tensor
    transmission: Tensor
    secondary_reflection: Tensor
    weights: TransportWeights


def generate_camera_rays(camera: Any, *, z_depth: bool = True) -> RayBundle:
    """Generate GLINT camera rays using its half-pixel sampling convention.

    When ``z_depth`` is true, directions have camera-space z equal to one. This
    is GLINT's default and makes ``origin + direction * depth`` compatible with
    the z-depth emitted by the interface rasterizer.
    """

    K = camera.K
    R = camera.R
    dtype, device = K.dtype, K.device
    height, width = int(camera.image_height), int(camera.image_width)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device) + 0.5,
        torch.arange(width, dtype=dtype, device=device) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    directions_camera = torch.einsum("ij,hwj->hwi", torch.linalg.inv(K), pixels)
    directions_world = torch.einsum("ij,hwj->hwi", R.T, directions_camera)
    if not z_depth:
        directions_world = F.normalize(directions_world, dim=-1)

    center = (-R.T @ camera.T.reshape(3, 1)).reshape(3)
    origins = center.reshape(1, 1, 3).expand(height, width, 3)
    return RayBundle(origins=origins, directions=directions_world)


def make_surface_rays(
    incoming: RayBundle,
    depth: Tensor,
    normal: Tensor,
    *,
    detach_depth: bool = True,
) -> tuple[RayBundle, RayBundle]:
    """Create GLINT's optically-thin transmission and reflection rays."""

    if depth.shape != incoming.origins.shape[:-1] + (1,):
        raise ValueError(
            f"depth must be [H, W, 1], got {tuple(depth.shape)} for rays "
            f"{tuple(incoming.origins.shape)}"
        )
    if normal.shape != incoming.origins.shape:
        raise ValueError(f"normal must be [H, W, 3], got {tuple(normal.shape)}")

    surface_depth = depth.detach() if detach_depth else depth
    origins = incoming.origins + incoming.directions * surface_depth
    unit_normal = F.normalize(normal, dim=-1)
    reflected_directions = (
        incoming.directions
        - 2.0
        * (incoming.directions * unit_normal).sum(dim=-1, keepdim=True)
        * unit_normal
    )
    return (
        RayBundle(origins=origins, directions=incoming.directions),
        RayBundle(origins=origins, directions=reflected_directions),
    )


def schlick_fresnel(
    view_directions: Tensor,
    normals: Tensor,
    *,
    f0: Union[float, Tensor] = 0.04,
) -> Tensor:
    """Evaluate GLINT Eq. (6) for incoming camera-ray directions."""

    outgoing = -F.normalize(view_directions, dim=-1)
    unit_normals = F.normalize(normals, dim=-1)
    cosine = (outgoing * unit_normals).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    f0_tensor = torch.as_tensor(f0, dtype=normals.dtype, device=normals.device)
    return f0_tensor + (1.0 - f0_tensor) * (1.0 - cosine).pow(5)


def compute_transport_weights(
    *,
    view_directions: Tensor,
    normals: Tensor,
    transparency: Tensor,
    specularity: Tensor,
    transmission_specularity: Tensor | None = None,
    f0: Union[float, Tensor] = 0.04,
) -> TransportWeights:
    """Compute the canonical three-path GLINT transport weights."""

    transparency = transparency.clamp(0.0, 1.0)
    specularity = specularity.clamp(0.0, 1.0)
    fresnel = schlick_fresnel(view_directions, normals, f0=f0).clamp(0.0, 1.0)
    reflection = (specularity + (1.0 - specularity) * fresnel).clamp(0.04, 1.0)
    diffuse = (1.0 - reflection) * (1.0 - transparency)
    transmission = (1.0 - reflection) * transparency
    secondary_reflection = torch.zeros_like(transmission)
    if transmission_specularity is not None:
        transmission_specularity = transmission_specularity.clamp(0.0, 1.0)
        secondary_reflection = transmission * transmission_specularity
        transmission = transmission * (1.0 - transmission_specularity)
    energy_sum = diffuse + reflection + transmission + secondary_reflection
    return TransportWeights(
        diffuse=diffuse,
        reflection=reflection,
        transmission=transmission,
        secondary_reflection=secondary_reflection,
        fresnel=fresnel,
        energy_sum=energy_sum,
    )


def compose_glint_radiance(
    *,
    interface_rgb: Tensor,
    reflection_rgb: Tensor,
    transmission_rgb: Tensor,
    view_directions: Tensor,
    normals: Tensor,
    transparency: Tensor,
    specularity: Tensor,
    secondary_reflection_rgb: Tensor | None = None,
    transmission_specularity: Tensor | None = None,
    f0: Union[float, Tensor] = 0.04,
) -> RadianceComposition:
    """Compose decomposed radiance according to the GLINT transport equation."""

    reference_shape = interface_rgb.shape
    for name, value in (
        ("reflection_rgb", reflection_rgb),
        ("transmission_rgb", transmission_rgb),
    ):
        if value.shape != reference_shape:
            raise ValueError(
                f"{name} must match interface RGB shape {reference_shape}, got "
                f"{tuple(value.shape)}"
            )
    if (secondary_reflection_rgb is None) != (transmission_specularity is None):
        raise ValueError(
            "secondary_reflection_rgb and transmission_specularity must be "
            "provided together"
        )
    if (
        secondary_reflection_rgb is not None
        and secondary_reflection_rgb.shape != reference_shape
    ):
        raise ValueError(
            "secondary_reflection_rgb must match interface RGB shape "
            f"{reference_shape}, got {tuple(secondary_reflection_rgb.shape)}"
        )

    weights = compute_transport_weights(
        view_directions=view_directions,
        normals=normals,
        transparency=transparency,
        specularity=specularity,
        transmission_specularity=transmission_specularity,
        f0=f0,
    )
    diffuse = weights.diffuse * interface_rgb
    reflection = weights.reflection * reflection_rgb
    transmission = weights.transmission * transmission_rgb
    secondary_reflection = (
        torch.zeros_like(interface_rgb)
        if secondary_reflection_rgb is None
        else weights.secondary_reflection * secondary_reflection_rgb
    )
    return RadianceComposition(
        rgb=diffuse + reflection + transmission + secondary_reflection,
        diffuse=diffuse,
        reflection=reflection,
        transmission=transmission,
        secondary_reflection=secondary_reflection,
        weights=weights,
    )


def _chw_to_hwc(value: Tensor) -> Tensor:
    return value.permute(1, 2, 0)


def _flatten_image(value: Tensor) -> Tensor:
    return value.reshape(1, -1, value.shape[-1])


def render_glint_transport(
    camera: Any,
    checkpoint: GlintCheckpoint,
    tracer: GlintRayTracer | None,
    *,
    stage: Literal["interface", "transmission", "full"] = "full",
    depth_ratio: float = 0.0,
    scaling_modifier: float = 1.0,
    f0: float = 0.04,
    detach_surface_depth: bool = True,
    render_transmission_direct: bool = False,
    render_secondary_reflection: bool = True,
) -> AttrDict:
    """Render one curriculum stage of the canonical GLINT pipeline."""

    if stage not in {"interface", "transmission", "full"}:
        raise ValueError(f"Unknown GLINT training stage: {stage}")
    if checkpoint.pcd is None:
        raise ValueError("GLINT transport requires the interface set")
    if stage != "interface" and (checkpoint.trans_env is None or tracer is None):
        raise ValueError(
            f"The {stage} stage requires transmission Gaussians and a tracer"
        )
    if stage == "full" and checkpoint.env is None:
        raise ValueError("The full stage requires reflection Gaussians")

    pipe = AttrDict(compute_cov3D_python=False, depth_ratio=depth_ratio)
    interaction = render_glint_camera(
        camera,
        checkpoint.pcd,
        pipe,
        checkpoint.bg_color,
        scaling_modifier=scaling_modifier,
        device=str(camera.K.device),
    )
    interface_rgb = _chw_to_hwc(interaction.render)
    depth = _chw_to_hwc(interaction.surf_depth)
    normals = _chw_to_hwc(interaction.rend_normal)
    transparency = _chw_to_hwc(interaction.transparency)
    material_transparency = _chw_to_hwc(interaction.conditional_transparency)
    specularity = _chw_to_hwc(interaction.specular)

    camera_rays = generate_camera_rays(camera, z_depth=True)
    transmission_rays, reflection_rays = make_surface_rays(
        camera_rays,
        depth,
        normals,
        detach_depth=detach_surface_depth,
    )
    zero_rgb = torch.zeros_like(interface_rgb)
    transmission_trace = None
    reflection_trace = None
    secondary_reflection_trace = None
    transmission_direct = None
    secondary_reflection = zero_rgb
    secondary_reflection_rays = None
    if stage == "interface":
        diffuse = interface_rgb
        reflection = zero_rgb
        transmission = zero_rgb
        fresnel = schlick_fresnel(camera_rays.directions, normals, f0=f0)
        energy_sum = torch.ones_like(transparency)
    else:
        assert tracer is not None and checkpoint.trans_env is not None
        need_transmission_direct = render_transmission_direct or (
            stage == "full" and render_secondary_reflection
        )
        if need_transmission_direct:
            # GLINT rasterizes G_trans from the target camera in addition to
            # tracing it along transmitted rays.  The direct render supplies
            # the normal/depth maps used by plausibility supervision and by
            # the original TRANS_NORMAL visualization.
            transmission_direct = render_glint_camera(
                camera,
                checkpoint.trans_env,
                pipe,
                checkpoint.trans_env_bg_color,
                scaling_modifier=scaling_modifier,
                device=str(camera.K.device),
            )
        transmission_trace = tracer.trace(
            checkpoint.trans_env,
            transmission_rays,
            background=checkpoint.trans_env_bg_color,
        )
        if stage == "transmission":
            # GLINT already reserves the Fresnel/specular reflection share
            # during transmission warm-up even though G_refl is not rendered
            # yet.  Re-normalizing the other two paths here creates a weight
            # discontinuity when the full stage begins.
            weights = compute_transport_weights(
                view_directions=camera_rays.directions,
                normals=normals,
                transparency=transparency,
                specularity=specularity,
                f0=f0,
            )
            diffuse = weights.diffuse * interface_rgb
            reflection = zero_rgb
            transmission = weights.transmission * transmission_trace.rgb
            fresnel = weights.fresnel
            energy_sum = weights.energy_sum
        else:
            assert checkpoint.env is not None
            reflection_trace = tracer.trace(
                checkpoint.env,
                reflection_rays,
                background=checkpoint.env_bg_color,
            )
            transmission_specularity = None
            if (
                render_secondary_reflection
                and transmission_direct is not None
                and transmission_trace.specular is not None
            ):
                direct_depth = _chw_to_hwc(transmission_direct.surf_depth)
                direct_normal = _chw_to_hwc(transmission_direct.rend_normal)
                _, secondary_reflection_rays = make_surface_rays(
                    camera_rays,
                    direct_depth,
                    direct_normal,
                    detach_depth=True,
                )
                secondary_reflection_trace = tracer.trace(
                    checkpoint.env,
                    secondary_reflection_rays,
                    background=checkpoint.env_bg_color,
                )
                transmission_specularity = transmission_trace.specular
            composition = compose_glint_radiance(
                interface_rgb=interface_rgb,
                reflection_rgb=reflection_trace.rgb,
                transmission_rgb=transmission_trace.rgb,
                view_directions=camera_rays.directions,
                normals=normals,
                transparency=transparency,
                specularity=specularity,
                secondary_reflection_rgb=(
                    None
                    if secondary_reflection_trace is None
                    else secondary_reflection_trace.rgb
                ),
                transmission_specularity=transmission_specularity,
                f0=f0,
            )
            diffuse = composition.diffuse
            reflection = composition.reflection
            transmission = composition.transmission
            secondary_reflection = composition.secondary_reflection
            fresnel = composition.weights.fresnel
            energy_sum = composition.weights.energy_sum

    rendered = diffuse + reflection + transmission + secondary_reflection

    output = AttrDict(
        stage=stage,
        render=rendered.permute(2, 0, 1),
        dif_render=diffuse.permute(2, 0, 1),
        ref_render=reflection.permute(2, 0, 1),
        trans_render=transmission.permute(2, 0, 1),
        secondary_ref_render=secondary_reflection.permute(2, 0, 1),
        rgb_map=_flatten_image(rendered),
        dif_rgb_map=_flatten_image(diffuse),
        ref_rgb_map=_flatten_image(reflection),
        trans_rgb_map=_flatten_image(transmission),
        secondary_ref_rgb_map=_flatten_image(secondary_reflection),
        acc_map=_flatten_image(_chw_to_hwc(interaction.rend_alpha)),
        dpt_map=_flatten_image(depth),
        norm_map=_flatten_image(normals),
        trans_map=_flatten_image(transparency),
        material_trans_map=_flatten_image(material_transparency),
        spec_map=_flatten_image(specularity),
        fresnel_refl=_flatten_image(fresnel),
        energy_sum=_flatten_image(energy_sum),
        ray_o=_flatten_image(camera_rays.origins),
        ray_d=_flatten_image(camera_rays.directions),
        ref_o=_flatten_image(reflection_rays.origins),
        ref_d=_flatten_image(reflection_rays.directions),
        trans_o=_flatten_image(transmission_rays.origins),
        trans_d=_flatten_image(transmission_rays.directions),
        secondary_ref_o=(
            None
            if secondary_reflection_rays is None
            else _flatten_image(secondary_reflection_rays.origins)
        ),
        secondary_ref_d=(
            None
            if secondary_reflection_rays is None
            else _flatten_image(secondary_reflection_rays.directions)
        ),
        interface=interaction,
        reflection_trace=reflection_trace,
        secondary_reflection_trace=secondary_reflection_trace,
        transmission_trace=transmission_trace,
        transmission_direct=transmission_direct,
    )
    return output
