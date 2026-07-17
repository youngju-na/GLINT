# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import torch

from glint.camera import GlintCamera
from glint.renderer import AttrDict
from glint.transport import TraceResult
from glint.visualizer import (
    GlintVisualizationConfig,
    GlintVisualizer,
    include_diagnostic_visualizations,
)


def _visualization_inputs() -> tuple[AttrDict, SimpleNamespace]:
    height, width = 6, 8
    camera = GlintCamera(
        K=torch.eye(3),
        R=torch.eye(3),
        T=torch.zeros(3, 1),
        image_width=width,
        image_height=height,
    )
    rgb = torch.linspace(0.0, 1.0, height * width * 3).reshape(height, width, 3)
    alpha = torch.full((height, width, 1), 0.8)
    depth = torch.linspace(1.0, 3.0, height * width).reshape(height, width, 1)
    normal = torch.zeros(height, width, 3)
    normal[..., 2] = -1.0
    trace = TraceResult(
        rgb=rgb.flip(-1),
        depth=depth + 1.0,
        alpha=alpha,
        normal=normal,
    )
    direct = AttrDict(
        render=(rgb * 0.9).permute(2, 0, 1),
        surf_depth=(depth + 2.0).permute(2, 0, 1),
        rend_alpha=(alpha * 0.5).permute(2, 0, 1),
        rend_normal=normal.permute(2, 0, 1),
    )
    output = AttrDict(
        render=rgb.permute(2, 0, 1),
        dif_render=(rgb * 0.5).permute(2, 0, 1),
        ref_render=(rgb * 0.2).permute(2, 0, 1),
        secondary_ref_render=(rgb * 0.05).permute(2, 0, 1),
        trans_render=(rgb * 0.3).permute(2, 0, 1),
        acc_map=alpha.reshape(1, -1, 1),
        dpt_map=depth.reshape(1, -1, 1),
        norm_map=normal.reshape(1, -1, 3),
        trans_map=torch.full((1, height * width, 1), 0.7),
        material_trans_map=torch.full((1, height * width, 1), 0.9),
        spec_map=torch.full((1, height * width, 1), 0.2),
        fresnel_refl=torch.full((1, height * width, 1), 0.04),
        energy_sum=torch.ones(1, height * width, 1),
        interface=AttrDict(surf_normal=normal.permute(2, 0, 1)),
        transmission_trace=trace,
        transmission_direct=direct,
        reflection_trace=trace,
        secondary_reflection_trace=trace,
    )
    sample = SimpleNamespace(
        camera=camera,
        rgb=rgb,
        normal=None,
        stable_normal=normal * 0.5 + 0.5,
        depth=depth,
    )
    return output, sample


def test_glint_visualizer_generates_original_and_trace_maps() -> None:
    output, sample = _visualization_inputs()
    visualizer = GlintVisualizer(GlintVisualizationConfig(normal_source="stable"))
    maps = visualizer.maps(output, sample)
    for name in (
        "RENDER",
        "DEPTH",
        "DEPTH_ALIGNED",
        "DEPTH_GT_SHARED",
        "DEPTH_ERROR",
        "NORMAL",
        "SURFACE_NORMAL",
        "SPECULAR",
        "TRANSPARENCY",
        "TRANSPARENCY_GATE",
        "DIFFUSE",
        "REFLECTION",
        "SECONDARY_REFLECTION",
        "TRANSMISSION",
        "ENV_RENDER",
        "TRANS_ENV_RENDER",
        "TRANS_DEPTH",
        "TRANS_NORMAL",
        "REFL_DEPTH",
        "REFL_NORMAL",
        "SECONDARY_ENV_RENDER",
    ):
        assert maps[name].shape == sample.rgb.shape
        assert torch.isfinite(maps[name]).all()
    assert torch.allclose(
        maps["TRANS_ENV_RENDER"],
        output.transmission_direct.render.permute(1, 2, 0),
    )
    assert torch.allclose(maps["TRANSPARENCY"], torch.full_like(sample.rgb, 0.9))
    assert torch.allclose(maps["TRANSPARENCY_GATE"], torch.full_like(sample.rgb, 0.7))
    # GLINT masks direct G_trans normals with the interface alpha, not the
    # transmission raster alpha.  This also verifies that the trace normal is
    # not used for the direct visualization path.
    expected_normal = torch.tensor([0.4, 0.4, 0.8]).expand_as(maps["TRANS_NORMAL"])
    assert torch.allclose(maps["TRANS_NORMAL"], expected_normal)
    assert "OPAQUE" not in maps

    output.confident_opaque_mask = torch.full((1, 6, 8, 1), 0.25)
    maps = visualizer.maps(output, sample)
    assert torch.allclose(maps["OPAQUE"], torch.full_like(maps["OPAQUE"], 0.25))


def test_legacy_visualization_list_keeps_port_diagnostics() -> None:
    names = include_diagnostic_visualizations(("render", "depth", "render"))
    assert names[:2] == ("RENDER", "DEPTH")
    assert len(names) == len(set(names))
    for diagnostic in (
        "DEPTH_ALIGNED",
        "DEPTH_ERROR",
        "SECONDARY_REFLECTION",
        "SECONDARY_ENV_RENDER",
        "TRANS_GUIDANCE_DEPTH_CONFIDENCE",
        "TRANS_GUIDANCE_DEPTH_EDGE",
        "TRANS_GUIDANCE_COVERAGE",
        "TRANSPARENCY_GATE",
    ):
        assert diagnostic in names


def test_glint_visualizer_saves_type_folders_gt_error_and_panel(
    tmp_path: Path,
) -> None:
    output, sample = _visualization_inputs()
    visualizer = GlintVisualizer(
        GlintVisualizationConfig(
            types=("RENDER", "DEPTH", "NORMAL", "TRANSPARENCY"),
            columns=3,
            normal_source="stable",
        )
    )
    paths = visualizer.save(output, sample, tmp_path, "frame0000_camera0000")
    assert paths["RENDER"] == tmp_path / "RENDER/frame0000_camera0000.png"
    assert paths["RENDER_GT"].name == "frame0000_camera0000_gt.png"
    assert paths["RENDER_ERROR"].name == "frame0000_camera0000_error.png"
    assert paths["NORMAL_GT"].parent.name == "NORMAL"
    assert paths["DEPTH_GT"].parent.name == "DEPTH"
    assert paths["PANEL"].exists()
    panel = iio.imread(paths["PANEL"])
    assert panel.ndim == 3 and panel.shape[-1] == 3
