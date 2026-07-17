# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""GLINT-style type-based visualization for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from torch import Tensor


DEFAULT_VISUALIZATION_TYPES = (
    "RENDER",
    "DEPTH",
    "DEPTH_ALIGNED",
    "DEPTH_GT_SHARED",
    "DEPTH_ERROR",
    "ALPHA",
    "NORMAL",
    "SURFACE_NORMAL",
    "SPECULAR",
    "TRANSPARENCY",
    "DIFFUSE",
    "REFLECTION",
    "SECONDARY_REFLECTION",
    "TRANSMISSION",
    "ENV_RENDER",
    "TRANS_ENV_RENDER",
    "TRANS_DEPTH",
    "TRANS_NORMAL",
    "SECONDARY_ENV_RENDER",
)

# Keep diagnostics introduced by the gsplat port even when a legacy GLINT YAML
# provides its older visualization list.  These maps are scene-independent and
# make depth alignment, transparency confidence, and the secondary transport
# path directly inspectable during training.
DIAGNOSTIC_VISUALIZATION_TYPES = (
    "DEPTH_ALIGNED",
    "DEPTH_GT_SHARED",
    "DEPTH_ERROR",
    "SECONDARY_REFLECTION",
    "SECONDARY_ENV_RENDER",
    "TRANS_GUIDANCE_DEPTH_CONFIDENCE",
    "TRANS_GUIDANCE_DEPTH_EDGE",
    "TRANS_GUIDANCE_COVERAGE",
    "TRANSPARENCY_GATE",
)


def include_diagnostic_visualizations(types: Sequence[str]) -> tuple[str, ...]:
    """Append port diagnostics without changing the configured map ordering."""

    return tuple(
        dict.fromkeys(
            [
                *(str(name).upper() for name in types),
                *DIAGNOSTIC_VISUALIZATION_TYPES,
            ]
        )
    )


@dataclass(frozen=True)
class GlintVisualizationConfig:
    types: tuple[str, ...] = DEFAULT_VISUALIZATION_TYPES
    depth_colormap: str = "turbo"
    depth_percentile: float = 0.01
    columns: int = 4
    save_individual: bool = True
    save_panel: bool = True
    normal_source: str = "diffren"

    def __post_init__(self) -> None:
        if not 0.0 <= self.depth_percentile < 0.5:
            raise ValueError("depth_percentile must be in [0, 0.5)")
        if self.columns <= 0:
            raise ValueError("visualization columns must be positive")
        if self.normal_source not in {"diffren", "stable"}:
            raise ValueError("normal_source must be 'diffren' or 'stable'")


def _flat_to_hwc(value: Tensor, height: int, width: int) -> Tensor:
    if value.ndim == 4 and value.shape[0] == 1 and value.shape[1:3] == (height, width):
        return value[0]
    if value.ndim == 3 and value.shape[:2] == (height, width):
        return value
    if value.ndim == 3 and value.shape[0] == 1:
        return value.reshape(height, width, value.shape[-1])
    if value.ndim == 3 and value.shape[1:] == (height, width):
        return value.permute(1, 2, 0)
    if value.ndim == 2 and value.shape[0] == height * width:
        return value.reshape(height, width, value.shape[-1])
    raise ValueError(
        f"Cannot convert tensor with shape {tuple(value.shape)} to {height}x{width} HWC"
    )


def _rgb(value: Tensor) -> Tensor:
    if value.shape[-1] == 3:
        return value
    if value.shape[-1] == 1:
        return value.expand(*value.shape[:-1], 3)
    raise ValueError(f"Visualization requires one or three channels, got {value.shape}")


def _view_normal(normal: Tensor, camera: Any, alpha: Tensor | None = None) -> Tensor:
    normal = F.normalize(normal, dim=-1) @ camera.R.T
    normal = normal.clone()
    normal[..., 1:] *= -1.0
    normal = normal * 0.5 + 0.5
    if alpha is not None:
        normal = normal * alpha
    return normal


def _target_normal(value: Tensor) -> Tensor:
    normal = F.normalize(value * 2.0 - 1.0, dim=-1)
    normal = normal.clone()
    normal[..., 1:] *= -1.0
    return normal * 0.5 + 0.5


def _depth_image(
    depth: Tensor,
    mask: Tensor,
    *,
    percentile: float,
    colormap_name: str,
    near: Tensor | None = None,
    far: Tensor | None = None,
) -> Tensor:
    depth = depth[..., :1]
    mask = mask[..., :1].bool() & torch.isfinite(depth) & (depth > 0)
    valid = depth[mask]
    normalized = torch.zeros_like(depth)
    if valid.numel():
        if near is None:
            near = torch.quantile(valid, percentile)
        if far is None:
            far = torch.quantile(valid, 1.0 - percentile)
        normalized = (1.0 - (depth - near) / (far - near).clamp_min(1e-6)).clamp(
            0.0, 1.0
        )
        normalized = normalized.masked_fill(~mask, 0.0)
    lut = torch.from_numpy(
        colormaps[colormap_name](np.linspace(0.0, 1.0, 256))[:, :3].astype(np.float32)
    ).to(device=depth.device, dtype=depth.dtype)
    indices = (normalized[..., 0] * 255.0).round().long()
    colored = lut[indices]
    return colored.masked_fill(~mask.expand_as(colored), 0.0)


def _depth_limits(
    depth: Tensor,
    mask: Tensor,
    percentile: float,
) -> tuple[Tensor, Tensor] | None:
    valid_mask = (
        mask[..., :1].bool() & torch.isfinite(depth[..., :1]) & (depth[..., :1] > 0)
    )
    valid = depth[..., :1][valid_mask]
    if not valid.numel():
        return None
    return (
        torch.quantile(valid, percentile),
        torch.quantile(valid, 1.0 - percentile),
    )


def _align_depth_to_target(
    prediction: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor]:
    prediction = prediction[..., :1]
    target = target[..., :1]
    mask = (
        torch.isfinite(prediction)
        & torch.isfinite(target)
        & (prediction > 0)
        & (target > 0)
    )
    weight = mask.to(prediction.dtype)
    a00 = (weight * prediction.square()).sum()
    a01 = (weight * prediction).sum()
    a11 = weight.sum()
    b0 = (weight * prediction * target).sum()
    b1 = (weight * target).sum()
    determinant = a00 * a11 - a01.square()
    if determinant.abs() <= 1e-8:
        return prediction, mask
    scale = (a11 * b0 - a01 * b1) / determinant
    shift = (-a01 * b0 + a00 * b1) / determinant
    return scale * prediction + shift, mask


def _depth_error_image(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    scale: Tensor,
) -> Tensor:
    error = (prediction[..., :1] - target[..., :1]).abs() / scale.clamp_min(1e-6)
    error = error.clamp(0.0, 1.0)
    lut = torch.from_numpy(
        colormaps["magma"](np.linspace(0.0, 1.0, 256))[:, :3].astype(np.float32)
    ).to(device=error.device, dtype=error.dtype)
    colored = lut[(error[..., 0] * 255.0).round().long()]
    valid = mask[..., :1].bool().expand_as(colored)
    return colored.masked_fill(~valid, 0.0)


def _to_uint8(image: Tensor) -> np.ndarray:
    return image.detach().clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()


def _titled_tile(name: str, image: Tensor) -> np.ndarray:
    array = _to_uint8(image)
    bar_height = max(22, array.shape[0] // 18)
    tile = np.zeros((array.shape[0] + bar_height, array.shape[1], 3), dtype=np.uint8)
    tile[bar_height:] = array
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    text_width = cv2.getTextSize(name, font, font_scale, 1)[0][0]
    if text_width > array.shape[1] - 10:
        font_scale *= (array.shape[1] - 10) / text_width
    cv2.putText(
        tile,
        name,
        (6, max(16, bar_height - 6)),
        font,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


class GlintVisualizer:
    """Generate the type folders and comparison panel used by GLINT."""

    def __init__(self, config: GlintVisualizationConfig | None = None) -> None:
        self.config = config or GlintVisualizationConfig()

    def maps(self, output: Any, sample: Any) -> dict[str, Tensor]:
        height = sample.camera.image_height
        width = sample.camera.image_width
        alpha = _flat_to_hwc(output.acc_map, height, width).clamp(0.0, 1.0)
        normal = _flat_to_hwc(output.norm_map, height, width)
        surface_normal = output.interface.surf_normal.permute(1, 2, 0)
        depth = _flat_to_hwc(output.dpt_map, height, width)
        transparency_gate = _flat_to_hwc(output.trans_map, height, width)
        transparency = _flat_to_hwc(
            getattr(output, "material_trans_map", output.trans_map),
            height,
            width,
        )

        maps: dict[str, Tensor] = {
            "RENDER": output.render.permute(1, 2, 0),
            "DEPTH": _depth_image(
                depth,
                alpha > 1e-4,
                percentile=self.config.depth_percentile,
                colormap_name=self.config.depth_colormap,
            ),
            "ALPHA": _rgb(alpha),
            "NORMAL": _view_normal(normal, sample.camera, alpha),
            "SURFACE_NORMAL": _view_normal(surface_normal, sample.camera, alpha),
            "SPECULAR": _rgb(_flat_to_hwc(output.spec_map, height, width)),
            "TRANSPARENCY": _rgb(transparency),
            "TRANSPARENCY_GATE": _rgb(transparency_gate),
            "DIFFUSE": output.dif_render.permute(1, 2, 0),
            "REFLECTION": output.ref_render.permute(1, 2, 0),
            "SECONDARY_REFLECTION": getattr(
                output,
                "secondary_ref_render",
                torch.zeros_like(output.ref_render),
            ).permute(1, 2, 0),
            "TRANSMISSION": output.trans_render.permute(1, 2, 0),
            "FRESNEL": _rgb(_flat_to_hwc(output.fresnel_refl, height, width)),
            "ENERGY_ERROR": _rgb(
                (_flat_to_hwc(output.energy_sum, height, width) - 1.0).abs()
            ),
        }
        if sample.depth is not None:
            target_depth = sample.depth
            aligned_depth, shared_mask = _align_depth_to_target(depth, target_depth)
            limits = _depth_limits(
                target_depth,
                shared_mask,
                self.config.depth_percentile,
            )
            if limits is not None:
                near, far = limits
                maps["DEPTH_ALIGNED"] = _depth_image(
                    aligned_depth,
                    shared_mask,
                    percentile=self.config.depth_percentile,
                    colormap_name=self.config.depth_colormap,
                    near=near,
                    far=far,
                )
                maps["DEPTH_GT_SHARED"] = _depth_image(
                    target_depth,
                    shared_mask,
                    percentile=self.config.depth_percentile,
                    colormap_name=self.config.depth_colormap,
                    near=near,
                    far=far,
                )
                maps["DEPTH_ERROR"] = _depth_error_image(
                    aligned_depth,
                    target_depth,
                    shared_mask,
                    scale=far - near,
                )
        confident_opaque = getattr(output, "confident_opaque_mask", None)
        if confident_opaque is not None:
            # The original OPAQUE visualization is the material/geometry
            # guidance label, not simply one minus the learned transparency.
            maps["OPAQUE"] = _rgb(_flat_to_hwc(confident_opaque, height, width))

        if output.transmission_trace is not None:
            trace = output.transmission_trace
            direct = getattr(output, "transmission_direct", None)
            trans_environment = (
                trace.rgb if direct is None else direct.render.permute(1, 2, 0)
            )
            trans_depth = (
                trace.depth if direct is None else direct.surf_depth.permute(1, 2, 0)
            )
            trans_alpha = (
                trace.alpha
                if direct is None
                else direct.rend_alpha.permute(1, 2, 0).clamp(0.0, 1.0)
            )
            trans_mask = (transparency > 0.5) & (trans_alpha > 1e-4)
            maps.update(
                {
                    "TRANS_ENV_RENDER": trans_environment,
                    "TRANS_DEPTH": _depth_image(
                        trans_depth,
                        trans_mask,
                        percentile=self.config.depth_percentile,
                        colormap_name=self.config.depth_colormap,
                    ),
                    "TRANS_ALPHA": _rgb(trans_alpha),
                }
            )
            if direct is not None:
                direct_normal = direct.rend_normal.permute(1, 2, 0)
                # Match GLINT's normal visualizer: normalize the direct G_trans
                # raster normal, then mask it by the interface alpha and the
                # transparent-material region.  Multiplying by G_trans alpha
                # here darkens the normal twice because rend_normal is already
                # an alpha-composited raster output.
                maps["TRANS_NORMAL"] = _view_normal(
                    direct_normal,
                    sample.camera,
                    alpha * (transparency > 0.5).to(alpha.dtype),
                )
            else:
                # Keep trace-only callers usable, but training/evaluation use
                # the direct rasterized map to match the released GLINT code.
                maps["TRANS_NORMAL"] = _view_normal(
                    trace.normal, sample.camera, trace.alpha
                )
        for name, key in (
            ("TRANS_GUIDANCE_ANGLE", "trans_guidance_angle_weight"),
            ("TRANS_GUIDANCE_DEPTH", "trans_guidance_depth_weight"),
            ("TRANS_GUIDANCE_WEIGHT", "trans_guidance_weight"),
            (
                "TRANS_GUIDANCE_DEPTH_CONFIDENCE",
                "trans_guidance_depth_confidence",
            ),
            ("TRANS_GUIDANCE_DEPTH_EDGE", "trans_guidance_depth_edge_weight"),
            ("TRANS_GUIDANCE_COVERAGE", "trans_guidance_coverage_weight"),
        ):
            value = getattr(output, key, None)
            if value is not None:
                maps[name] = _rgb(_flat_to_hwc(value, height, width))
        if output.reflection_trace is not None:
            trace = output.reflection_trace
            reflection_mask = trace.alpha > 1e-4
            maps.update(
                {
                    "ENV_RENDER": trace.rgb,
                    "REFL_DEPTH": _depth_image(
                        trace.depth,
                        reflection_mask,
                        percentile=self.config.depth_percentile,
                        colormap_name=self.config.depth_colormap,
                    ),
                    "REFL_ALPHA": _rgb(trace.alpha),
                    "REFL_NORMAL": _view_normal(
                        trace.normal, sample.camera, _rgb(trace.alpha)[..., :1]
                    ),
                }
            )
        secondary_trace = getattr(output, "secondary_reflection_trace", None)
        if secondary_trace is not None:
            maps["SECONDARY_ENV_RENDER"] = secondary_trace.rgb
        return maps

    def save(
        self,
        output: Any,
        sample: Any,
        output_dir: str | Path,
        stem: str,
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        available = self.maps(output, sample)
        names = [name.upper() for name in self.config.types]

        prediction = available["RENDER"]
        target = sample.rgb
        error = 3.0 * (prediction - target).square().sum(dim=-1, keepdim=True)
        available["RENDER_GT"] = target
        available["RENDER_ERROR"] = _rgb(error.clamp(0.0, 1.0))
        selected: dict[str, Tensor] = {
            "RENDER_GT": available["RENDER_GT"],
            "RENDER": prediction,
            "RENDER_ERROR": available["RENDER_ERROR"],
        }
        normal_target = (
            sample.stable_normal
            if self.config.normal_source == "stable"
            else sample.normal
        )
        if normal_target is not None:
            available["NORMAL_GT"] = _target_normal(normal_target)
        if sample.depth is not None:
            target_depth = sample.depth
            available["DEPTH_GT"] = _depth_image(
                target_depth,
                target_depth > 0,
                percentile=self.config.depth_percentile,
                colormap_name=self.config.depth_colormap,
            )
        for name in names:
            if name == "RENDER" or name not in available:
                continue
            selected[name] = available[name]
            if name == "DEPTH" and "DEPTH_GT" in available:
                selected["DEPTH_GT"] = available["DEPTH_GT"]
            if name == "NORMAL" and "NORMAL_GT" in available:
                selected["NORMAL_GT"] = available["NORMAL_GT"]

        paths: dict[str, Path] = {}
        if self.config.save_individual:
            for name, image in selected.items():
                suffix = ""
                folder = name
                if name == "RENDER_GT":
                    folder, suffix = "RENDER", "_gt"
                elif name == "RENDER_ERROR":
                    folder, suffix = "RENDER", "_error"
                elif name == "NORMAL_GT":
                    folder, suffix = "NORMAL", "_gt"
                elif name == "DEPTH_GT":
                    folder, suffix = "DEPTH", "_gt"
                path = output_dir / folder / f"{stem}{suffix}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(path, _to_uint8(image))
                paths[name] = path

        if self.config.save_panel and selected:
            tiles = [_titled_tile(name, image) for name, image in selected.items()]
            columns = min(self.config.columns, len(tiles))
            rows = (len(tiles) + columns - 1) // columns
            tile_height = max(tile.shape[0] for tile in tiles)
            tile_width = max(tile.shape[1] for tile in tiles)
            panel = np.zeros(
                (rows * tile_height, columns * tile_width, 3), dtype=np.uint8
            )
            for index, tile in enumerate(tiles):
                row, column = divmod(index, columns)
                panel[
                    row * tile_height : row * tile_height + tile.shape[0],
                    column * tile_width : column * tile_width + tile.shape[1],
                ] = tile
            path = output_dir / "PANELS" / f"{stem}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(path, panel)
            paths["PANEL"] = path
        return paths
