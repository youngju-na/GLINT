# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""GLINT helpers built on gsplat's native 2DGS renderer.

Public symbols are imported lazily so that camera, dataset, and config tools do
not initialize CUDA or build gsplat extensions merely by importing this package.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "AttrDict": ("renderer", "AttrDict"),
    "DEFAULT_VISUALIZATION_TYPES": ("visualizer", "DEFAULT_VISUALIZATION_TYPES"),
    "GlintBatch": ("dataset", "GlintBatch"),
    "GlintCamera": ("camera", "GlintCamera"),
    "GlintCheckpoint": ("checkpoint", "GlintCheckpoint"),
    "GlintDataset": ("dataset", "GlintDataset"),
    "GlintFrame": ("dataset", "GlintFrame"),
    "GlintGaussianSet": ("checkpoint", "GlintGaussianSet"),
    "GlintRayTracer": ("transport", "GlintRayTracer"),
    "GlintRefiner": ("refinement", "GlintRefiner"),
    "GlintSample": ("dataset", "GlintSample"),
    "GlintStageSchedule": ("model", "GlintStageSchedule"),
    "GlintTrainingModel": ("model", "GlintTrainingModel"),
    "GlintVisualizationConfig": ("visualizer", "GlintVisualizationConfig"),
    "GlintVisualizer": ("visualizer", "GlintVisualizer"),
    "OptixSurfelTracer": ("optix_backend", "OptixSurfelTracer"),
    "RadianceComposition": ("transport", "RadianceComposition"),
    "RayBundle": ("transport", "RayBundle"),
    "RefinementConfig": ("refinement", "RefinementConfig"),
    "TorchSurfelTracer": ("torch_tracer", "TorchSurfelTracer"),
    "TraceResult": ("transport", "TraceResult"),
    "TrainingStage": ("model", "TrainingStage"),
    "TransportWeights": ("transport", "TransportWeights"),
    "collate_glint_samples": ("dataset", "collate_glint_samples"),
    "compose_glint_radiance": ("transport", "compose_glint_radiance"),
    "compute_transport_weights": ("transport", "compute_transport_weights"),
    "create_gaussian_set": ("model", "create_gaussian_set"),
    "decode_diffren_depth": ("dataset", "decode_diffren_depth"),
    "generate_camera_rays": ("transport", "generate_camera_rays"),
    "list_easyvolcap_cameras": ("camera", "list_easyvolcap_cameras"),
    "load_colmap_ply": ("model", "load_colmap_ply"),
    "load_easyvolcap_camera": ("camera", "load_easyvolcap_camera"),
    "load_easyvolcap_cameras": ("camera", "load_easyvolcap_cameras"),
    "load_easyvolcap_config": ("dataset", "load_easyvolcap_config"),
    "load_glint_checkpoint": ("checkpoint", "load_glint_checkpoint"),
    "make_glint_dataloader": ("dataset", "make_glint_dataloader"),
    "make_surface_rays": ("transport", "make_surface_rays"),
    "make_surfel_triangles": ("optix_backend", "make_surfel_triangles"),
    "quaternion_to_rotation_matrix": (
        "geometry",
        "quaternion_to_rotation_matrix",
    ),
    "rasterize_glint_2dgs": ("renderer", "rasterize_glint_2dgs"),
    "render_glint_camera": ("renderer", "render_glint_camera"),
    "render_glint_transport": ("transport", "render_glint_transport"),
    "schlick_fresnel": ("transport", "schlick_fresnel"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f".{module_name}", __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
