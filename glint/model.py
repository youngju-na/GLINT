# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Lean role-specific Gaussian model used by the standalone GLINT trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree
from torch import Tensor, nn

from glint.checkpoint import (
    GlintCheckpoint,
    GlintGaussianRole,
    GlintGaussianSet,
)


TrainingStage = Literal["interface", "transmission", "full"]
_SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class GlintStageSchedule:
    """Iteration boundaries for GLINT's three-set curriculum."""

    transmission_start: int = 1_000
    reflection_start: int = 3_000
    interface_freeze: int | None = None

    def __post_init__(self) -> None:
        if self.transmission_start < 0:
            raise ValueError("transmission_start must be non-negative")
        if self.reflection_start < self.transmission_start:
            raise ValueError("reflection_start must not precede transmission_start")
        if self.interface_freeze is not None and self.interface_freeze < 0:
            raise ValueError("interface_freeze must be non-negative or None")

    def stage(self, step: int) -> TrainingStage:
        if step < self.transmission_start:
            return "interface"
        if step < self.reflection_start:
            return "transmission"
        return "full"

    def train_interface(self, step: int) -> bool:
        return self.interface_freeze is None or step < self.interface_freeze


def load_colmap_ply(
    path: str | Path,
    *,
    max_points: int | None = None,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """Read xyz and uint8 RGB from a COLMAP/EasyVolCap point-cloud PLY."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    vertex = PlyData.read(str(path))["vertex"]
    names = {property.name for property in vertex.properties}
    required = {"x", "y", "z", "red", "green", "blue"}
    missing = required - names
    if missing:
        raise ValueError(f"{path} is missing PLY properties {sorted(missing)}")
    xyz = np.stack(
        (np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])),
        axis=-1,
    ).astype(np.float32)
    rgb = np.stack(
        (
            np.asarray(vertex["red"]),
            np.asarray(vertex["green"]),
            np.asarray(vertex["blue"]),
        ),
        axis=-1,
    ).astype(np.float32)
    rgb /= 255.0
    if max_points is not None and len(xyz) > max_points:
        if max_points <= 3:
            raise ValueError("max_points must be greater than three")
        generator = np.random.default_rng(seed)
        selected = np.sort(generator.choice(len(xyz), size=max_points, replace=False))
        xyz, rgb = xyz[selected], rgb[selected]
    return torch.from_numpy(xyz), torch.from_numpy(rgb)


def _initial_log_scales(xyz: Tensor, neighbors: int = 3) -> Tensor:
    """Initialize two tangent scales using an efficient CPU k-d tree."""

    points = xyz.detach().cpu().numpy()
    if len(points) <= neighbors:
        raise ValueError(f"Need more than {neighbors} points to initialize scales")
    distances, _ = cKDTree(points).query(points, k=neighbors + 1, workers=-1)
    rms = np.sqrt(np.mean(np.square(distances[:, 1:]), axis=-1))
    rms = np.maximum(rms, 1e-7).astype(np.float32)
    return torch.from_numpy(np.log(rms))[:, None].repeat(1, 2)


def create_gaussian_set(
    xyz: Tensor,
    rgb: Tensor,
    *,
    role: GlintGaussianRole,
    sh_degree: int = 3,
    init_opacity: float = 0.1,
    init_specular: float = 1e-3,
    init_transparency: float = 1e-2,
    device: str | torch.device = "cuda",
) -> GlintGaussianSet:
    """Create one independently trainable Gaussian set from colored points."""

    if xyz.shape != rgb.shape or xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(
            f"xyz and rgb must both be [N, 3], got {xyz.shape}, {rgb.shape}"
        )
    if not 0.0 < init_opacity < 1.0:
        raise ValueError("init_opacity must be in (0, 1)")
    n = xyz.shape[0]
    coefficients = (sh_degree + 1) ** 2
    features_dc = ((rgb.float() - 0.5) / _SH_C0).reshape(n, 1, 3)
    features_rest = torch.zeros(n, coefficients - 1, 3, dtype=torch.float32)
    quaternions = torch.randn(n, 4, dtype=torch.float32)
    quaternions = torch.nn.functional.normalize(quaternions, dim=-1)
    state: dict[str, Tensor] = {
        "set.active_sh_degree": torch.zeros(1, dtype=torch.long),
        "set._xyz": xyz.float(),
        "set._features_dc": features_dc,
        "set._features_rest": features_rest,
        "set._scaling": _initial_log_scales(xyz),
        "set._rotation": quaternions,
        "set._opacity": torch.full((n, 1), init_opacity).logit(),
    }
    if role in {"interface", "transmission"}:
        if not 0.0 < init_specular < 1.0:
            raise ValueError("Specular initializer must be in (0, 1)")
        state["set._specular"] = torch.full((n, 1), init_specular).logit()
    if role == "interface":
        if not 0.0 < init_transparency < 1.0:
            raise ValueError("Transparency initializer must be in (0, 1)")
        state["set._transmission_coeff"] = torch.full((n, 1), init_transparency).logit()
    return GlintGaussianSet(
        state,
        "set",
        role=role,
        trainable=True,
    ).to(device)


class GlintTrainingModel(nn.Module):
    """Three independent Gaussian sets for interface, transmission, reflection."""

    def __init__(
        self,
        interface: GlintGaussianSet,
        transmission: GlintGaussianSet,
        reflection: GlintGaussianSet,
    ) -> None:
        super().__init__()
        if interface.role != "interface":
            raise ValueError("interface set has the wrong role")
        if transmission.role != "transmission":
            raise ValueError("transmission set has the wrong role")
        if reflection.role != "reflection":
            raise ValueError("reflection set has the wrong role")
        self.interface = interface
        self.transmission = transmission
        self.reflection = reflection
        device = interface.get_xyz.device
        self.register_buffer("interface_background", torch.zeros(3, device=device))
        self.register_buffer("transmission_background", torch.zeros(3, device=device))
        self.register_buffer("reflection_background", torch.zeros(3, device=device))

    @classmethod
    def from_ply(
        cls,
        interface_ply: str | Path,
        reflection_ply: str | Path,
        *,
        sh_degree: int = 3,
        environment_sh_degree: int | None = None,
        max_interface_points: int | None = None,
        max_reflection_points: int | None = None,
        device: str | torch.device = "cuda",
        seed: int = 42,
    ) -> "GlintTrainingModel":
        interface_xyz, interface_rgb = load_colmap_ply(
            interface_ply, max_points=max_interface_points, seed=seed
        )
        reflection_xyz, reflection_rgb = load_colmap_ply(
            reflection_ply, max_points=max_reflection_points, seed=seed
        )
        if environment_sh_degree is None:
            environment_sh_degree = sh_degree
        # GLINT initializes G_trans from the same SfM points as G_intr, but the
        # tensors are cloned so the two sets evolve independently.
        return cls(
            interface=create_gaussian_set(
                interface_xyz,
                interface_rgb,
                role="interface",
                sh_degree=sh_degree,
                device=device,
            ),
            transmission=create_gaussian_set(
                interface_xyz.clone(),
                interface_rgb.clone(),
                role="transmission",
                sh_degree=environment_sh_degree,
                device=device,
            ),
            reflection=create_gaussian_set(
                reflection_xyz,
                reflection_rgb,
                role="reflection",
                sh_degree=environment_sh_degree,
                device=device,
            ),
        )

    def as_checkpoint(self, step: int | None = None) -> GlintCheckpoint:
        return GlintCheckpoint(
            pcd=self.interface,
            env=self.reflection,
            trans_env=self.transmission,
            bg_color=self.interface_background,
            env_bg_color=self.reflection_background,
            trans_env_bg_color=self.transmission_background,
            epoch=step,
        )

    def legacy_state_dict(self) -> dict[str, Tensor]:
        """Export a lean state readable by ``load_glint_checkpoint``."""

        output: dict[str, Tensor] = {
            "sampler.bg_color": self.interface_background,
            "sampler.env_bg_color": self.reflection_background,
            "sampler.trans_env_bg_color": self.transmission_background,
        }
        for gaussian_set, prefix in (
            (self.interface, "sampler.pcd"),
            (self.reflection, "sampler.env"),
            (self.transmission, "sampler.trans_env"),
        ):
            output[f"{prefix}.active_sh_degree"] = gaussian_set.active_sh_degree
            for standard, parameter in gaussian_set.parameter_map().items():
                raw_name = gaussian_set._STANDARD_NAMES[standard]
                output[f"{prefix}.{raw_name}"] = parameter
        return output
