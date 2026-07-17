# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Load GLINT/EasyVolCap checkpoints without importing EasyVolCap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Literal, Mapping, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


GlintGaussianRole = Literal["interface", "transmission", "reflection"]


class GlintGaussianSet(nn.Module):
    """Role-specific GLINT Gaussian parameters and their activations.

    The interface stores specularity and transparency.  The transmission set
    stores specularity because GLINT uses the first transmitted hit to weight
    secondary reflection.  The reflection set needs only geometry, opacity,
    and radiance.  Legacy roughness and IOR tensors remain intentionally
    unused.
    """

    _BASE_PARAMETER_NAMES = (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
    )
    _INTERFACE_PARAMETER_NAMES = ("_specular", "_transmission_coeff")
    _TRANSMISSION_PARAMETER_NAMES = ("_specular",)
    _STANDARD_NAMES = {
        "means": "_xyz",
        "sh0": "_features_dc",
        "shN": "_features_rest",
        "scales": "_scaling",
        "quats": "_rotation",
        "opacities": "_opacity",
        "specular": "_specular",
        "transparency": "_transmission_coeff",
    }

    def __init__(
        self,
        state: Mapping[str, Tensor],
        prefix: str,
        *,
        role: GlintGaussianRole | None = None,
        render_reflection: bool | None = None,
        trainable: bool,
    ) -> None:
        super().__init__()
        if role is None:
            role = "interface" if render_reflection else "reflection"
        if role not in {"interface", "transmission", "reflection"}:
            raise ValueError(f"Unknown GLINT Gaussian role: {role}")
        names = self._BASE_PARAMETER_NAMES
        if role == "interface":
            names += self._INTERFACE_PARAMETER_NAMES
        elif role == "transmission":
            names += self._TRANSMISSION_PARAMETER_NAMES
        for name in names:
            key = f"{prefix}.{name}"
            if key not in state:
                # Checkpoints produced by the first lean gsplat port omitted
                # G_trans specularity.  Restore the released GLINT initializer
                # so those checkpoints remain loadable.
                if role == "transmission" and name == "_specular":
                    value = torch.full_like(state[f"{prefix}._opacity"], 1e-3).logit()
                else:
                    raise KeyError(f"GLINT checkpoint is missing {key}")
            else:
                value = state[key]
            setattr(
                self,
                name,
                nn.Parameter(value.detach().clone(), requires_grad=trainable),
            )
        active_key = f"{prefix}.active_sh_degree"
        if active_key not in state:
            raise KeyError(f"GLINT checkpoint is missing {active_key}")
        self.register_buffer(
            "active_sh_degree", state[active_key].detach().clone().reshape(1)
        )
        self.role: GlintGaussianRole = role
        self.render_reflection = role == "interface"
        self.has_specular = hasattr(self, "_specular")
        self.has_transparency = hasattr(self, "_transmission_coeff")
        self.specular_channels = (
            self._specular.shape[-1] if self.render_reflection else 0
        )

    @property
    def get_xyz(self) -> Tensor:
        return self._xyz

    @property
    def get_features(self) -> Tensor:
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_scaling(self) -> Tensor:
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> Tensor:
        return F.normalize(self._rotation, dim=-1)

    @property
    def get_opacity(self) -> Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def get_specular(self) -> Tensor:
        if not self.has_specular:
            raise AttributeError(f"{self.role} Gaussians do not have specularity")
        return torch.sigmoid(self._specular)

    @property
    def get_transmission_coeff(self) -> Tensor:
        if not self.has_transparency:
            raise AttributeError(f"{self.role} Gaussians do not have transparency")
        return torch.sigmoid(self._transmission_coeff)

    def parameter_map(self) -> dict[str, nn.Parameter]:
        """Return gsplat-strategy names for all parameters present on this role."""

        return {
            standard: getattr(self, raw)
            for standard, raw in self._STANDARD_NAMES.items()
            if hasattr(self, raw)
        }

    def replace_parameter_map(self, parameters: Mapping[str, nn.Parameter]) -> None:
        """Synchronize parameters replaced by a densification operation."""

        for standard, parameter in parameters.items():
            setattr(self, self._STANDARD_NAMES[standard], parameter)


@dataclass
class GlintCheckpoint:
    pcd: Optional[GlintGaussianSet]
    env: Optional[GlintGaussianSet]
    trans_env: Optional[GlintGaussianSet]
    bg_color: Tensor
    env_bg_color: Tensor
    trans_env_bg_color: Tensor
    epoch: Optional[int]


def load_glint_checkpoint(
    path: Union[str, Path],
    *,
    gaussian_sets: Collection[str] = ("pcd", "env", "trans_env"),
    device: Union[str, torch.device] = "cpu",
    trainable: bool = False,
    trust_checkpoint: bool = False,
) -> GlintCheckpoint:
    """Load selected GLINT Gaussian sets from an EasyVolCap ``.pt`` checkpoint.

    Old EasyVolCap checkpoints can contain NumPy objects in optimizer state that
    PyTorch's safe weights-only loader rejects. Set ``trust_checkpoint=True``
    only for a checkpoint whose origin you trust; it enables regular pickle
    loading even though this adapter reads only the model tensors.
    """

    requested = set(gaussian_sets)
    unknown = requested - {"pcd", "env", "trans_env"}
    if unknown:
        raise ValueError(f"Unknown GLINT Gaussian sets: {sorted(unknown)}")

    payload = torch.load(
        Path(path), map_location="cpu", weights_only=not trust_checkpoint
    )
    state = payload["model"] if "model" in payload else payload

    def build(name: str, role: GlintGaussianRole) -> Optional[GlintGaussianSet]:
        if name not in requested:
            return None
        return GlintGaussianSet(
            state,
            f"sampler.{name}",
            role=role,
            trainable=trainable,
        ).to(device)

    def background(key: str) -> Tensor:
        if key not in state:
            raise KeyError(f"GLINT checkpoint is missing {key}")
        return state[key].detach().clone().to(device)

    return GlintCheckpoint(
        pcd=build("pcd", "interface"),
        env=build("env", "reflection"),
        trans_env=build("trans_env", "transmission"),
        bg_color=background("sampler.bg_color"),
        env_bg_color=background("sampler.env_bg_color"),
        trans_env_bg_color=background("sampler.trans_env_bg_color"),
        epoch=payload.get("epoch") if isinstance(payload, Mapping) else None,
    )
