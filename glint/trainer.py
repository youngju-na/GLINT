# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Standalone three-stage trainer for canonical GLINT radiance transport."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from glint.dataset import GlintBatch, GlintDataset, make_glint_dataloader
from glint.losses import ssim_loss
from glint.model import GlintStageSchedule, GlintTrainingModel, TrainingStage
from glint.optix_backend import OptixSurfelTracer
from glint.refinement import GlintRefiner, RefinementConfig
from glint.renderer import AttrDict, render_glint_camera
from glint.transport import generate_camera_rays, render_glint_transport
from glint.visualizer import GlintVisualizer


@dataclass(frozen=True)
class GlintLossConfig:
    rgb_l1: float = 1.2
    rgb_ssim: float = 0.4
    normal_prior: float = 0.3
    normal_source: str = "diffren"
    normal_prior_start: int = 0
    normal_prior_stop: int | None = None
    normal_cos_threshold_step: int = 10_000
    normal_cos_threshold_initial: float = 0.5
    normal_cos_threshold_final: float = 0.9
    normal_cos_threshold_final_step: int = 31_000
    depth_normal_thresholding: bool = True
    normal_depth_weight: bool = True
    depth_prior: float = 0.1
    depth_prior_start: int = 0
    depth_prior_stop: int | None = None
    normal_consistency: float = 0.1
    normal_consistency_start: int = 0
    normal_consistency_stop: int | None = None
    normal_consistency_depth_weight: bool = True
    normal_smooth: float = 0.05
    normal_smooth_start: int = 5_000
    transparency_regularization: float = 0.01
    transparency_regularization_start: int = 7_000
    transparency_guidance: float = 0.01
    transparency_guidance_start: int = 5_000
    transparency_guidance_stop: int | None = None
    transmission_opacity: float = 0.01
    transmission_opacity_start: int = 7_000
    transmission_opacity_stop: int | None = 31_000
    albedo_threshold: float = 0.05
    basecolor_threshold: float = 0.25
    depth_discrepancy_threshold: float = 0.005
    depth_discrepancy_relative_threshold: float = 0.005
    depth_discrepancy_margin_ratio: float = 0.5
    depth_discrepancy_softness_ratio: float = 0.25
    transparency_depth_edge_scale: float = 0.02
    transparency_label_smoothing: float = 0.02
    transparency_positive_weight: float = 1.0
    transparency_negative_weight: float = 1.0
    transparency_mask_shrink: int = 3
    transparency_cleanup_start: int = 21_000
    transparency_cleanup_final_step: int = 31_000
    transparency_cleanup_opaque_multiplier: float = 3.0
    transparency_primitive_entropy_weight: float = 1.0
    transparency_cleanup_transparent_min: float = 0.8
    transparency_cleanup_transparent_weight: float = 1.0
    transparency_angle_weighting: bool = True
    transparency_angle_soft_start: float = 70.0
    transparency_angle_hard_cutoff: float = 80.0
    transparency_far_depth_quantile: float = 1.0
    transparency_explainability_gating: bool = True
    transparency_explainability_start: int | None = None
    transparency_explainability_patch: int = 5
    transparency_explainability_threshold: float = 0.02
    transparency_explainability_sharpness: float = 0.01
    transparency_explainability_min_weight: float = 0.1
    plausibility: float = 0.05
    plausibility_start: int = 5_000
    plausibility_stop: int | None = None
    perceptual: float = 0.05
    perceptual_start: int = 15_000
    perceptual_every: int = 1
    perceptual_net: str = "vgg"
    multi_view: float = 0.01
    multi_view_start: int = 10_000
    multi_view_stop: int | None = 21_000
    multi_view_geo: float = 0.03
    multi_view_ncc: float = 0.15
    multi_view_every: int = 10
    multi_view_patch_size: int = 2
    multi_view_pixel_noise: float = 1.0
    multi_view_samples: int = 51_200

    def __post_init__(self) -> None:
        if self.normal_source not in {"diffren", "stable"}:
            raise ValueError(
                "normal_source must be either 'diffren' or 'stable', got "
                f"{self.normal_source!r}"
            )
        if self.perceptual_every <= 0 or self.multi_view_every <= 0:
            raise ValueError("loss sampling intervals must be positive")
        if self.transparency_mask_shrink <= 0:
            raise ValueError("transparency_mask_shrink must be positive")
        if self.transparency_cleanup_final_step < self.transparency_cleanup_start:
            raise ValueError(
                "transparency_cleanup_final_step must not precede "
                "transparency_cleanup_start"
            )
        if self.transparency_cleanup_opaque_multiplier < 1.0:
            raise ValueError(
                "transparency_cleanup_opaque_multiplier must be at least 1"
            )
        if self.transparency_primitive_entropy_weight < 0.0:
            raise ValueError(
                "transparency_primitive_entropy_weight must be non-negative"
            )
        if not 0.0 <= self.transparency_cleanup_transparent_min <= 1.0:
            raise ValueError("transparency_cleanup_transparent_min must be in [0, 1]")
        if self.transparency_cleanup_transparent_weight < 0.0:
            raise ValueError(
                "transparency_cleanup_transparent_weight must be non-negative"
            )
        if self.depth_discrepancy_threshold < 0:
            raise ValueError("depth_discrepancy_threshold must be non-negative")
        if self.depth_discrepancy_relative_threshold < 0:
            raise ValueError(
                "depth_discrepancy_relative_threshold must be non-negative"
            )
        if self.depth_discrepancy_margin_ratio < 0:
            raise ValueError("depth_discrepancy_margin_ratio must be non-negative")
        if self.depth_discrepancy_softness_ratio <= 0:
            raise ValueError("depth_discrepancy_softness_ratio must be positive")
        if self.transparency_depth_edge_scale <= 0:
            raise ValueError("transparency_depth_edge_scale must be positive")
        if not -1.0 <= self.normal_cos_threshold_initial <= 1.0:
            raise ValueError("normal_cos_threshold_initial must be in [-1, 1]")
        if not -1.0 <= self.normal_cos_threshold_final <= 1.0:
            raise ValueError("normal_cos_threshold_final must be in [-1, 1]")
        if self.normal_cos_threshold_final < self.normal_cos_threshold_initial:
            raise ValueError(
                "normal_cos_threshold_final must be at least as strict as "
                "normal_cos_threshold_initial"
            )
        if self.normal_cos_threshold_final_step < self.normal_cos_threshold_step:
            raise ValueError(
                "normal_cos_threshold_final_step must not precede "
                "normal_cos_threshold_step"
            )

    def normal_cosine_threshold(self, step: int) -> float | None:
        """Return the scheduled normal-confidence cutoff for one training step."""

        if step < self.normal_cos_threshold_step:
            return None
        duration = self.normal_cos_threshold_final_step - self.normal_cos_threshold_step
        if duration <= 0:
            return self.normal_cos_threshold_final
        progress = min(
            max((step - self.normal_cos_threshold_step) / duration, 0.0),
            1.0,
        )
        return self.normal_cos_threshold_initial + progress * (
            self.normal_cos_threshold_final - self.normal_cos_threshold_initial
        )

    def transparency_cleanup_progress(self, step: int) -> float:
        """Ramp confidence-aware cleanup only after topology refinement settles."""

        if step < self.transparency_cleanup_start:
            return 0.0
        duration = (
            self.transparency_cleanup_final_step - self.transparency_cleanup_start
        )
        if duration <= 0:
            return 1.0
        return min(
            max((step - self.transparency_cleanup_start) / duration, 0.0),
            1.0,
        )

    @classmethod
    def from_easyvolcap(cls, config: Mapping[str, Any]) -> "GlintLossConfig":
        supervisor = config.get("model_cfg", {}).get("supervisor_cfg", {})

        def optional_int(name: str, default: int | None) -> int | None:
            value = supervisor.get(name, default)
            return None if value is None else int(value)

        return cls(
            rgb_l1=float(supervisor.get("img_loss_weight", cls.rgb_l1)),
            rgb_ssim=float(supervisor.get("ssim_loss_weight", cls.rgb_ssim)),
            normal_prior=float(supervisor.get("norm_loss_weight", cls.normal_prior)),
            normal_source=str(supervisor.get("use_normal_type", cls.normal_source)),
            normal_prior_start=int(
                supervisor.get("norm_loss_start_iter", cls.normal_prior_start)
            ),
            normal_prior_stop=optional_int(
                "norm_loss_until_iter", cls.normal_prior_stop
            ),
            normal_cos_threshold_step=int(
                supervisor.get(
                    "normal_cos_threshold_iter", cls.normal_cos_threshold_step
                )
            ),
            normal_cos_threshold_initial=float(
                supervisor.get(
                    "normal_cos_threshold_initial",
                    cls.normal_cos_threshold_initial,
                )
            ),
            normal_cos_threshold_final=float(
                supervisor.get(
                    "normal_cos_threshold_final",
                    cls.normal_cos_threshold_final,
                )
            ),
            normal_cos_threshold_final_step=int(
                supervisor.get(
                    "normal_cos_threshold_final_iter",
                    cls.normal_cos_threshold_final_step,
                )
            ),
            depth_normal_thresholding=bool(
                supervisor.get(
                    "use_normal_threshold_for_depth_loss",
                    cls.depth_normal_thresholding,
                )
            ),
            normal_depth_weight=bool(
                supervisor.get("use_dpt_scale_norm_loss", cls.normal_depth_weight)
            ),
            depth_prior=float(supervisor.get("dpt_loss_weight", cls.depth_prior)),
            depth_prior_start=int(
                supervisor.get("dpt_loss_start_iter", cls.depth_prior_start)
            ),
            depth_prior_stop=optional_int("dpt_loss_until_iter", cls.depth_prior_stop),
            normal_consistency=float(
                supervisor.get("gs_norm_loss_weight", cls.normal_consistency)
            ),
            normal_consistency_start=int(
                supervisor.get("gs_norm_loss_start_iter", cls.normal_consistency_start)
            ),
            normal_consistency_stop=optional_int(
                "gs_norm_loss_until_iter", cls.normal_consistency_stop
            ),
            normal_consistency_depth_weight=bool(
                supervisor.get(
                    "use_dpt_scale_gs_norm_loss",
                    cls.normal_consistency_depth_weight,
                )
            ),
            normal_smooth=float(
                supervisor.get("norm_smooth_loss_weight", cls.normal_smooth)
            ),
            normal_smooth_start=int(
                supervisor.get("norm_smooth_loss_start_iter", cls.normal_smooth_start)
            ),
            transparency_regularization=float(
                supervisor.get(
                    "trans_map_reg_loss_weight", cls.transparency_regularization
                )
            ),
            transparency_regularization_start=int(
                supervisor.get(
                    "trans_reg_loss_start_iter",
                    cls.transparency_regularization_start,
                )
            ),
            transparency_guidance=float(
                supervisor.get("trans_guidance_loss_weight", cls.transparency_guidance)
            ),
            transparency_guidance_start=int(
                supervisor.get(
                    "trans_guidance_loss_start_iter",
                    cls.transparency_guidance_start,
                )
            ),
            transparency_guidance_stop=optional_int(
                "trans_guidance_loss_until_iter", cls.transparency_guidance_stop
            ),
            transmission_opacity=float(
                supervisor.get(
                    "trans_env_opacity_loss_weight", cls.transmission_opacity
                )
            ),
            transmission_opacity_start=int(
                supervisor.get(
                    "trans_env_opacity_loss_start_iter",
                    cls.transmission_opacity_start,
                )
            ),
            transmission_opacity_stop=optional_int(
                "trans_env_opacity_loss_until_iter",
                cls.transmission_opacity_stop,
            ),
            albedo_threshold=float(
                supervisor.get("albedo_threshold", cls.albedo_threshold)
            ),
            basecolor_threshold=float(
                supervisor.get("basecolor_threshold", cls.basecolor_threshold)
            ),
            depth_discrepancy_threshold=float(
                supervisor.get(
                    "depth_discrepancy_threshold", cls.depth_discrepancy_threshold
                )
            ),
            depth_discrepancy_relative_threshold=float(
                supervisor.get(
                    "depth_discrepancy_relative_threshold",
                    cls.depth_discrepancy_relative_threshold,
                )
            ),
            depth_discrepancy_margin_ratio=float(
                supervisor.get(
                    "depth_discrepancy_margin_ratio",
                    cls.depth_discrepancy_margin_ratio,
                )
            ),
            depth_discrepancy_softness_ratio=float(
                supervisor.get(
                    "depth_discrepancy_softness_ratio",
                    cls.depth_discrepancy_softness_ratio,
                )
            ),
            transparency_depth_edge_scale=float(
                supervisor.get(
                    "transparency_depth_edge_scale",
                    cls.transparency_depth_edge_scale,
                )
            ),
            transparency_label_smoothing=float(
                supervisor.get(
                    "trans_label_smoothing", cls.transparency_label_smoothing
                )
            ),
            transparency_positive_weight=float(
                supervisor.get("trans_pos_weight", cls.transparency_positive_weight)
            ),
            transparency_negative_weight=float(
                supervisor.get("trans_neg_weight", cls.transparency_negative_weight)
            ),
            transparency_mask_shrink=int(
                supervisor.get("trans_mask_shrink_kernel", cls.transparency_mask_shrink)
            ),
            transparency_cleanup_start=int(
                supervisor.get(
                    "trans_cleanup_start_iter", cls.transparency_cleanup_start
                )
            ),
            transparency_cleanup_final_step=int(
                supervisor.get(
                    "trans_cleanup_final_iter",
                    cls.transparency_cleanup_final_step,
                )
            ),
            transparency_cleanup_opaque_multiplier=float(
                supervisor.get(
                    "trans_cleanup_opaque_multiplier",
                    cls.transparency_cleanup_opaque_multiplier,
                )
            ),
            transparency_primitive_entropy_weight=float(
                supervisor.get(
                    "trans_primitive_entropy_weight",
                    cls.transparency_primitive_entropy_weight,
                )
            ),
            transparency_cleanup_transparent_min=float(
                supervisor.get(
                    "trans_cleanup_transparent_min",
                    cls.transparency_cleanup_transparent_min,
                )
            ),
            transparency_cleanup_transparent_weight=float(
                supervisor.get(
                    "trans_cleanup_transparent_weight",
                    cls.transparency_cleanup_transparent_weight,
                )
            ),
            transparency_angle_weighting=bool(
                supervisor.get(
                    "use_view_depth_aware_trans_guidance_weighting",
                    cls.transparency_angle_weighting,
                )
            ),
            transparency_angle_soft_start=float(
                supervisor.get(
                    "trans_angle_soft_start_deg",
                    cls.transparency_angle_soft_start,
                )
            ),
            transparency_angle_hard_cutoff=float(
                supervisor.get(
                    "trans_angle_hard_cutoff_deg",
                    cls.transparency_angle_hard_cutoff,
                )
            ),
            transparency_far_depth_quantile=float(
                supervisor.get(
                    "trans_guidance_far_depth_quantile",
                    cls.transparency_far_depth_quantile,
                )
            ),
            transparency_explainability_gating=bool(
                supervisor.get(
                    "use_opaque_explainability_trans_guidance_gating",
                    cls.transparency_explainability_gating,
                )
            ),
            transparency_explainability_start=optional_int(
                "trans_opaque_explainability_gating_start_iter",
                cls.transparency_explainability_start,
            ),
            transparency_explainability_patch=int(
                supervisor.get(
                    "trans_opaque_explainability_patch_size",
                    cls.transparency_explainability_patch,
                )
            ),
            transparency_explainability_threshold=float(
                supervisor.get(
                    "trans_opaque_explainability_threshold",
                    cls.transparency_explainability_threshold,
                )
            ),
            transparency_explainability_sharpness=float(
                supervisor.get(
                    "trans_opaque_explainability_sharpness",
                    cls.transparency_explainability_sharpness,
                )
            ),
            transparency_explainability_min_weight=float(
                supervisor.get(
                    "trans_opaque_explainability_min_weight",
                    cls.transparency_explainability_min_weight,
                )
            ),
            plausibility=float(
                supervisor.get("plausibility_loss_weight", cls.plausibility)
            ),
            plausibility_start=int(
                supervisor.get("plausibility_loss_start_iter", cls.plausibility_start)
            ),
            plausibility_stop=optional_int(
                "plausibility_loss_end_iter", cls.plausibility_stop
            ),
            perceptual=float(supervisor.get("perc_loss_weight", cls.perceptual)),
            perceptual_start=int(
                supervisor.get("perc_loss_start_iter", cls.perceptual_start)
            ),
            perceptual_every=int(
                supervisor.get("perc_loss_every_n_iter", cls.perceptual_every)
            ),
            multi_view=float(supervisor.get("multi_view_loss_weight", cls.multi_view)),
            multi_view_start=int(
                supervisor.get("multi_view_start_iter", cls.multi_view_start)
            ),
            multi_view_stop=optional_int("multi_view_until_iter", cls.multi_view_stop),
            multi_view_geo=float(
                supervisor.get("multi_view_geo_weight", cls.multi_view_geo)
            ),
            multi_view_ncc=float(
                supervisor.get("multi_view_ncc_weight", cls.multi_view_ncc)
            ),
            multi_view_every=int(
                config.get("model_cfg", {})
                .get("sampler_cfg", {})
                .get("multi_view_every_n_iter", cls.multi_view_every)
            ),
            multi_view_patch_size=int(
                config.get("model_cfg", {})
                .get("sampler_cfg", {})
                .get("multi_view_patch_size", cls.multi_view_patch_size)
            ),
            multi_view_pixel_noise=float(
                config.get("model_cfg", {})
                .get("sampler_cfg", {})
                .get("multi_view_pixel_noise_th", cls.multi_view_pixel_noise)
            ),
            multi_view_samples=int(
                config.get("model_cfg", {})
                .get("sampler_cfg", {})
                .get("multi_view_sample_num", cls.multi_view_samples)
            ),
        )


@dataclass(frozen=True)
class GlintTrainerConfig:
    max_steps: int = 60_000
    sh_degree_interval: int = 1_000
    interface_sh_start: int = 0
    environment_sh_start: int = 0
    position_lr_max_steps: int = 30_000
    interface_geometry_freeze_step: int | None = None
    log_every: int = 10
    save_every: int = 5_000
    image_every: int = 1_000
    num_workers: int = 4
    seed: int = 42
    scene_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.sh_degree_interval <= 0:
            raise ValueError("sh_degree_interval must be positive")
        if self.position_lr_max_steps <= 0:
            raise ValueError("position_lr_max_steps must be positive")
        if (
            self.interface_geometry_freeze_step is not None
            and self.interface_geometry_freeze_step < 0
        ):
            raise ValueError("interface_geometry_freeze_step must be non-negative")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")


def _make_optimizers(
    model: GlintTrainingModel,
    *,
    scene_scale: float,
) -> dict[str, dict[str, torch.optim.Optimizer]]:
    learning_rates = {
        "means": 1.6e-4 * scene_scale,
        "sh0": 2.5e-3,
        "shN": 1.25e-4,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "specular": 1e-2,
        "transparency": 5e-2,
    }
    output: dict[str, dict[str, torch.optim.Optimizer]] = {}
    for name, gaussian_set in (
        ("interface", model.interface),
        ("transmission", model.transmission),
        ("reflection", model.reflection),
    ):
        output[name] = {
            parameter_name: torch.optim.Adam(
                [parameter],
                lr=learning_rates[parameter_name],
                eps=1e-15,
            )
            for parameter_name, parameter in gaussian_set.parameter_map().items()
        }
    return output


_INTERFACE_GEOMETRY_PARAMETERS = frozenset({"means", "scales", "quats", "opacities"})


def _compute_scale_and_shift(
    prediction: Tensor, target: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor]:
    a00 = (mask * prediction.square()).sum(dim=(1, 2))
    a01 = (mask * prediction).sum(dim=(1, 2))
    a11 = mask.sum(dim=(1, 2))
    b0 = (mask * prediction * target).sum(dim=(1, 2))
    b1 = (mask * target).sum(dim=(1, 2))
    determinant = a00 * a11 - a01.square()
    valid = determinant.abs() > 1e-8
    scale = torch.zeros_like(b0)
    shift = torch.zeros_like(b1)
    scale[valid] = (a11[valid] * b0[valid] - a01[valid] * b1[valid]) / determinant[
        valid
    ]
    shift[valid] = (-a01[valid] * b0[valid] + a00[valid] * b1[valid]) / determinant[
        valid
    ]
    return scale, shift


def _scale_shift_invariant_depth_loss(
    prediction: Tensor, target: Tensor, mask: Tensor | None = None
) -> Tensor:
    prediction = prediction[..., 0]
    target = target[..., 0]
    if mask is None:
        mask = target > 0
    elif mask.ndim == 4:
        mask = mask[..., 0]
    mask = mask.to(dtype=prediction.dtype) * (target > 0).to(prediction.dtype)
    scale, shift = _compute_scale_and_shift(prediction, target, mask)
    aligned = scale[:, None, None] * prediction + shift[:, None, None]
    denominator = mask.sum().clamp_min(1.0)
    data = (mask * (aligned - target).square()).sum() / (2.0 * denominator)
    gradient = aligned - target
    regularization = prediction.new_tensor(0.0)
    for stride in (1, 2, 4, 8):
        value = gradient[:, ::stride, ::stride]
        value_mask = mask[:, ::stride, ::stride]
        dx = (value[:, :, 1:] - value[:, :, :-1]).abs()
        dy = (value[:, 1:, :] - value[:, :-1, :]).abs()
        mx = value_mask[:, :, 1:] * value_mask[:, :, :-1]
        my = value_mask[:, 1:, :] * value_mask[:, :-1, :]
        regularization = (
            regularization + ((dx * mx).sum() + (dy * my).sum()) / denominator
        )
    return data + 0.5 * regularization


def _edge_aware_smoothness(value: Tensor, rgb: Tensor) -> Tensor:
    value_dx = value[:, :, 1:] - value[:, :, :-1]
    value_dy = value[:, 1:] - value[:, :-1]
    rgb_dx = rgb[:, :, 1:] - rgb[:, :, :-1]
    rgb_dy = rgb[:, 1:] - rgb[:, :-1]
    weight_x = torch.exp(-rgb_dx.abs().mean(dim=-1, keepdim=True))
    weight_y = torch.exp(-rgb_dy.abs().mean(dim=-1, keepdim=True))
    return (weight_x * value_dx.abs()).mean() + (weight_y * value_dy.abs()).mean()


def _inverse_depth_weight(depth: Tensor, percentile: float = 0.01) -> Tensor:
    """Near-to-far [1, 0] weighting used by the released GLINT losses."""

    depth = depth.detach()
    flat = depth.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros_like(depth)
    count = max(int(flat.numel() * percentile), 1)
    near = flat.topk(count, largest=False).values.max()
    far = flat.topk(count, largest=True).values.min()
    return (1.0 - (depth - near) / (far - near).clamp_min(1e-6)).clamp(0.0, 1.0)


def _weighted_mean(value: Tensor, weight: Tensor) -> Tensor:
    weight = weight.to(dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _binary_entropy(probability: Tensor) -> Tensor:
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    return -(
        probability * probability.log()
        + (1.0 - probability) * (1.0 - probability).log()
    )


def _disjoint_transparency_masks(
    transparent: Tensor,
    opaque: Tensor,
) -> tuple[Tensor, Tensor]:
    """Cancel shared confidence where transparent/opaque pseudo-labels conflict."""

    overlap = torch.minimum(transparent, opaque)
    return (transparent - overlap).clamp_min(0.0), (opaque - overlap).clamp_min(0.0)


def _confidence_aware_transparency_regularization(
    transparency: Tensor,
    target: Tensor,
    transparent_mask: Tensor | None,
    opaque_mask: Tensor | None,
    config: GlintLossConfig,
    step: int,
    *,
    primitive_transparency: Tensor | None = None,
    primitive_weight: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    """Sharpen material routing without allowing image entropy to move geometry."""

    if transparent_mask is None or opaque_mask is None:
        zero = torch.zeros_like(transparency)
        transparent_mask = zero
        opaque_mask = zero
    else:
        transparent_mask = transparent_mask.detach().to(transparency.dtype)
        opaque_mask = opaque_mask.detach().to(transparency.dtype)

    confidence = torch.maximum(transparent_mask, opaque_mask)
    pixel_entropy = _weighted_mean(_binary_entropy(transparency), confidence)
    opaque_mean = _weighted_mean(transparency, opaque_mask)

    progress = config.transparency_cleanup_progress(step)
    primitive_entropy = transparency.new_tensor(0.0)
    if primitive_transparency is not None and primitive_weight is not None:
        primitive_entropy = _weighted_mean(
            _binary_entropy(primitive_transparency),
            primitive_weight.detach(),
        )
    entropy = (1.0 - progress) * pixel_entropy + progress * (
        config.transparency_primitive_entropy_weight * primitive_entropy
    )
    opaque_multiplier = 1.0 + progress * (
        config.transparency_cleanup_opaque_multiplier - 1.0
    )
    transparent_protection = transparency.new_tensor(0.0)
    if progress > 0.0 and config.transparency_cleanup_transparent_weight > 0.0:
        transparent_floor = F.relu(
            config.transparency_cleanup_transparent_min - transparency
        ).square()
        transparent_protection = _weighted_mean(transparent_floor, transparent_mask)

    loss = (
        entropy
        + opaque_multiplier * opaque_mean
        + progress
        * config.transparency_cleanup_transparent_weight
        * transparent_protection
        + _edge_aware_smoothness(transparency, target)
    )
    diagnostics: dict[str, Tensor | float] = {
        "transparency_cleanup_progress": progress,
        "transparency_cleanup_opaque_multiplier": opaque_multiplier,
        "transparency_pixel_entropy": pixel_entropy,
        "transparency_primitive_entropy": primitive_entropy,
    }
    return loss, diagnostics


def _normal_agreement(
    prediction: Tensor,
    target: Tensor,
    threshold: float | None,
) -> tuple[Tensor, Tensor]:
    """Return cosine agreement and a detached confidence mask."""

    valid = torch.isfinite(prediction).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    cosine = (prediction * target).sum(dim=-1).clamp(-1.0, 1.0)
    cosine = torch.where(valid, cosine, torch.zeros_like(cosine))
    if threshold is not None:
        # ``valid`` is saved by the preceding torch.where for cosine's
        # backward.  Mutating it in place invalidates autograd exactly when
        # the scheduled threshold activates (step 10k in the released config).
        valid = valid & (cosine.detach() > threshold)
    return cosine, valid


def _shrink_mask(mask: Tensor, kernel: int) -> Tensor:
    if kernel <= 1:
        return mask
    if kernel % 2 == 0:
        kernel += 1
    nchw = mask.permute(0, 3, 1, 2)
    eroded = -F.max_pool2d(-nchw, kernel, stride=1, padding=kernel // 2)
    return eroded.permute(0, 2, 3, 1)


def _depth_edge_confidence(depth: Tensor, scale: float) -> Tensor:
    """Downweight discontinuities using a local, relative depth gradient."""

    value = depth.permute(0, 3, 1, 2)
    dx = (value[..., 1:] - value[..., :-1]).abs()
    dy = (value[..., 1:, :] - value[..., :-1, :]).abs()
    edge_x = torch.maximum(F.pad(dx, (0, 1)), F.pad(dx, (1, 0)))
    edge_y = torch.maximum(F.pad(dy, (0, 0, 0, 1)), F.pad(dy, (0, 0, 1, 0)))
    relative_edge = torch.maximum(edge_x, edge_y) / value.abs().clamp_min(1e-3)
    return torch.exp(-relative_edge / scale).permute(0, 2, 3, 1)


def _depth_discrepancy_confidence(
    interface_depth: Tensor,
    direct_depth: Tensor,
    config: GlintLossConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build soft transparent/opaque confidence with a relative dead band."""

    depth_gap = direct_depth - interface_depth
    local_depth_scale = torch.maximum(
        interface_depth.abs(), direct_depth.abs()
    ).clamp_min(1e-3)
    discrepancy_threshold = torch.maximum(
        torch.full_like(depth_gap, config.depth_discrepancy_threshold),
        config.depth_discrepancy_relative_threshold * local_depth_scale,
    )
    margin = config.depth_discrepancy_margin_ratio * discrepancy_threshold
    softness = (
        config.depth_discrepancy_softness_ratio * discrepancy_threshold
    ).clamp_min(1e-6)
    transparent = torch.sigmoid((depth_gap - discrepancy_threshold - margin) / softness)
    opaque = torch.sigmoid((discrepancy_threshold - margin - depth_gap) / softness)
    edge = _depth_edge_confidence(
        interface_depth,
        config.transparency_depth_edge_scale,
    )
    return transparent, opaque, edge


def _project_world(points: Tensor, camera: Any) -> tuple[Tensor, Tensor]:
    camera_points = torch.einsum("ij,...j->...i", camera.R, points) + camera.T.reshape(
        3
    )
    projected = torch.einsum("ij,...j->...i", camera.K, camera_points)
    pixels = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
    return pixels, camera_points[..., 2:]


def _sample_image(image: Tensor, pixels: Tensor) -> Tensor:
    """Bilinearly sample an HWC image at arbitrary pixel coordinates."""

    height, width = image.shape[:2]
    grid = pixels.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
    sampled = F.grid_sample(
        image.permute(2, 0, 1).unsqueeze(0),
        grid.reshape(1, -1, 1, 2),
        align_corners=True,
        padding_mode="border",
    )
    return sampled.reshape(image.shape[-1], -1).T.reshape(*pixels.shape[:-1], -1)


def _local_ncc(reference: Tensor, neighbor: Tensor) -> tuple[Tensor, Tensor]:
    reference = reference - reference.mean(dim=-1, keepdim=True)
    neighbor = neighbor - neighbor.mean(dim=-1, keepdim=True)
    numerator = (reference * neighbor).sum(dim=-1).square()
    denominator = reference.square().sum(dim=-1) * neighbor.square().sum(dim=-1) + 1e-8
    loss = (1.0 - numerator / denominator).clamp(0.0, 2.0)
    return loss, loss < 0.9


class GlintTrainer:
    """Own optimizers, curriculum, refinement, loss, and checkpoint lifecycle."""

    def __init__(
        self,
        model: GlintTrainingModel,
        dataset: GlintDataset,
        *,
        output_dir: str | Path,
        schedule: GlintStageSchedule,
        loss_config: GlintLossConfig,
        trainer_config: GlintTrainerConfig,
        refinement_configs: Mapping[str, RefinementConfig] | None = None,
        visualizer: GlintVisualizer | None = None,
        start_step: int = 0,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.schedule = schedule
        self.loss_config = loss_config
        self.config = trainer_config
        self.step = start_step
        self.device = model.interface.get_xyz.device
        self.tracer = OptixSurfelTracer()
        self.tracer.train()
        self._lpips_model: torch.nn.Module | None = None
        self._interface_geometry_frozen = False
        self.visualizer = visualizer or GlintVisualizer()
        self.optimizers = _make_optimizers(
            model,
            scene_scale=trainer_config.scene_scale,
        )
        configs = refinement_configs or {
            "interface": RefinementConfig(),
            "transmission": RefinementConfig(),
            "reflection": RefinementConfig(),
        }
        self.refiners = {
            "interface": GlintRefiner(
                model.interface,
                self.optimizers["interface"],
                configs["interface"],
                scene_scale=trainer_config.scene_scale,
            ),
            "transmission": GlintRefiner(
                model.transmission,
                self.optimizers["transmission"],
                configs["transmission"],
                scene_scale=trainer_config.scene_scale,
            ),
            "reflection": GlintRefiner(
                model.reflection,
                self.optimizers["reflection"],
                configs["reflection"],
                scene_scale=trainer_config.scene_scale,
            ),
        }
        # Sanitize initialization and resumed checkpoints before their first
        # render.  This catches isolated-point KNN scale outliers as well as
        # covariance corruption in checkpoints whose interface is already
        # frozen and would otherwise never enter refinement again.
        initial_covariance_stats = {
            name: refiner.stabilize_covariance()
            for name, refiner in self.refiners.items()
        }
        if any(
            int(stats["scale_clamped"]) > 0
            for stats in initial_covariance_stats.values()
        ):
            print(f"initial_covariance_stabilization={initial_covariance_stats}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "config.json").write_text(
            json.dumps(
                {
                    "schedule": asdict(schedule),
                    "loss": asdict(loss_config),
                    "trainer": asdict(trainer_config),
                    "refinement": {
                        name: asdict(value) for name, value in configs.items()
                    },
                    "initial_covariance_stabilization": initial_covariance_stats,
                    "visualization": asdict(self.visualizer.config),
                    "dataset": dataset.summary(),
                },
                indent=2,
            )
            + "\n"
        )

    def _active_sets(self, step: int) -> tuple[str, ...]:
        active: list[str] = []
        if self.schedule.train_interface(step):
            active.append("interface")
        if step >= self.schedule.transmission_start:
            active.append("transmission")
        if step >= self.schedule.reflection_start:
            active.append("reflection")
        return tuple(active)

    def _zero_grad(self) -> None:
        for optimizers in self.optimizers.values():
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)

    def _freeze_interface_geometry_if_needed(self, step: int) -> None:
        freeze_step = self.config.interface_geometry_freeze_step
        if self._interface_geometry_frozen or freeze_step is None or step < freeze_step:
            return
        parameters = self.model.interface.parameter_map()
        for name in _INTERFACE_GEOMETRY_PARAMETERS:
            parameters[name].requires_grad_(False)
        self._interface_geometry_frozen = True
        print(
            "[freeze] "
            f"step={step} interface_geometry={sorted(_INTERFACE_GEOMETRY_PARAMETERS)}"
        )

    def _update_learning_rates(self, step: int) -> None:
        progress = min(step / self.config.position_lr_max_steps, 1.0)
        gamma = 0.01**progress
        base = 1.6e-4 * self.config.scene_scale
        for optimizers in self.optimizers.values():
            optimizers["means"].param_groups[0]["lr"] = base * gamma

    def _update_sh_degree(self, step: int) -> None:
        starts = {
            "interface": self.config.interface_sh_start,
            "transmission": self.config.environment_sh_start,
            "reflection": self.config.environment_sh_start,
        }
        for name, gaussian_set in (
            ("interface", self.model.interface),
            ("transmission", self.model.transmission),
            ("reflection", self.model.reflection),
        ):
            age = max(0, step - starts[name])
            maximum = int(math.sqrt(gaussian_set.get_features.shape[1]) - 1)
            degree = min(age // self.config.sh_degree_interval, maximum)
            gaussian_set.active_sh_degree.fill_(degree)

    def _perceptual_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self._lpips_model is None:
            try:
                import lpips
            except ImportError as error:
                raise RuntimeError(
                    "LPIPS is enabled by the GLINT config. Install examples/requirements.txt "
                    "in the splat environment before reaching perceptual_start."
                ) from error
            model = lpips.LPIPS(
                net=self.loss_config.perceptual_net,
                verbose=False,
            ).to(self.device)
            model.eval()
            model.requires_grad_(False)
            self._lpips_model = model
        return self._lpips_model(
            prediction.permute(0, 3, 1, 2),
            target.permute(0, 3, 1, 2),
        ).mean()

    def _transparency_guidance(
        self,
        output: Any,
        sample: Any,
        transparency: Tensor,
        target: Tensor,
        step: int,
    ) -> Tensor | None:
        config = self.loss_config
        direct = getattr(output, "transmission_direct", None)
        if direct is None or sample.diffuse_albedo is None or sample.basecolor is None:
            return None

        height, width = sample.camera.image_height, sample.camera.image_width
        interface_depth = output.dpt_map.reshape(1, height, width, 1).detach()
        coverage_map = getattr(output, "acc_map", None)
        coverage_weight = (
            torch.ones_like(interface_depth)
            if coverage_map is None
            else coverage_map.reshape(1, height, width, 1).detach().clamp(0.0, 1.0)
        )
        direct_depth = direct.surf_depth.permute(1, 2, 0).unsqueeze(0).detach()
        albedo = sample.diffuse_albedo.unsqueeze(0).mean(dim=-1, keepdim=True)
        basecolor = sample.basecolor.unsqueeze(0).mean(dim=-1, keepdim=True)
        low_albedo = albedo < config.albedo_threshold
        high_albedo = albedo > config.albedo_threshold + 0.05
        high_basecolor = basecolor > config.basecolor_threshold
        depth_exists = (interface_depth > 0.0) & (direct_depth > 0.0)
        (
            transparent_depth_confidence,
            opaque_depth_confidence,
            depth_edge_weight,
        ) = _depth_discrepancy_confidence(
            interface_depth,
            direct_depth,
            config,
        )
        if sample.sky_mask is None:
            not_sky = torch.ones_like(depth_exists)
        else:
            not_sky = sample.sky_mask.unsqueeze(0) < 0.5
        sky = ~not_sky

        transparent_material = low_albedo & high_basecolor & not_sky & depth_exists
        material_opaque = high_albedo | (low_albedo & ~high_basecolor) | sky
        transparent_mask = (
            transparent_material.to(transparency.dtype)
            * transparent_depth_confidence
            * depth_edge_weight
            * coverage_weight
        )
        geometry_opaque = (
            depth_exists.to(transparency.dtype)
            * opaque_depth_confidence
            * depth_edge_weight
            * coverage_weight
        )
        opaque_mask = torch.maximum(
            material_opaque.to(transparency.dtype) * coverage_weight,
            geometry_opaque,
        )
        output.confident_transparent_mask = transparent_mask
        output.confident_opaque_mask = opaque_mask
        transparent_mask = _shrink_mask(
            transparent_mask, config.transparency_mask_shrink
        )
        opaque_mask = _shrink_mask(opaque_mask, config.transparency_mask_shrink)

        normal_world = output.norm_map.reshape(1, height, width, 3).detach()
        ray_direction = output.ray_d.reshape(1, height, width, 3).detach()
        cosine = (
            (F.normalize(normal_world, dim=-1) * -F.normalize(ray_direction, dim=-1))
            .sum(dim=-1, keepdim=True)
            .abs()
        )
        soft = math.cos(math.radians(config.transparency_angle_soft_start))
        hard = math.cos(math.radians(config.transparency_angle_hard_cutoff))
        angle_weight = ((cosine - hard) / max(soft - hard, 1e-6)).clamp(0.0, 1.0)

        depth_weight = torch.ones_like(transparency)
        quantile = config.transparency_far_depth_quantile
        if 0.0 < quantile < 1.0:
            valid_depth = interface_depth > 0
            if valid_depth.any():
                far = torch.quantile(interface_depth[valid_depth], quantile)
                depth_weight = ((interface_depth <= far) | ~valid_depth).to(
                    transparency.dtype
                )
        guidance_weight = (
            angle_weight * depth_weight
            if config.transparency_angle_weighting
            else torch.ones_like(transparency)
        )

        explainability = torch.ones_like(transparency)
        explainability_start = config.transparency_explainability_start
        if explainability_start is None:
            explainability_start = config.transparency_guidance_start
        if config.transparency_explainability_gating and step >= explainability_start:
            with torch.no_grad():
                opaque_rgb = (
                    output.dif_render.permute(1, 2, 0)
                    + output.ref_render.permute(1, 2, 0)
                    + output.secondary_ref_render.permute(1, 2, 0)
                ).unsqueeze(0)
                full_rgb = output.render.permute(1, 2, 0).unsqueeze(0)
                opaque_error = (target - opaque_rgb).abs().mean(dim=-1, keepdim=True)
                full_error = (target - full_rgb).abs().mean(dim=-1, keepdim=True)
                benefit = (opaque_error - full_error).clamp_min(0.0)
                patch = max(1, config.transparency_explainability_patch)
                if patch % 2 == 0:
                    patch += 1
                if patch > 1:
                    benefit = F.avg_pool2d(
                        benefit.permute(0, 3, 1, 2),
                        patch,
                        stride=1,
                        padding=patch // 2,
                    ).permute(0, 2, 3, 1)
                sharpness = max(config.transparency_explainability_sharpness, 1e-6)
                explainability = torch.sigmoid(
                    (benefit - config.transparency_explainability_threshold) / sharpness
                )
                minimum = config.transparency_explainability_min_weight
                explainability = minimum + (1.0 - minimum) * explainability

        transparent_mask = transparent_mask * guidance_weight * explainability
        opaque_mask = opaque_mask * guidance_weight
        transparent_mask, opaque_mask = _disjoint_transparency_masks(
            transparent_mask,
            opaque_mask,
        )
        # Reuse the exact detached guidance masks for late cleanup.  This keeps
        # the stricter regularizer to cheap elementwise reductions and avoids
        # a second material/depth pass.
        output.transparency_positive_mask = transparent_mask
        output.transparency_negative_mask = opaque_mask
        smoothing = config.transparency_label_smoothing
        positive_target = torch.full_like(transparency, 1.0 - smoothing)
        negative_target = torch.full_like(transparency, smoothing)
        positive_bce = F.binary_cross_entropy(
            transparency, positive_target, reduction="none"
        )
        negative_bce = F.binary_cross_entropy(
            transparency, negative_target, reduction="none"
        )
        loss = config.transparency_positive_weight * _weighted_mean(
            positive_bce, transparent_mask
        )
        loss = loss + config.transparency_negative_weight * _weighted_mean(
            negative_bce, opaque_mask
        )

        if sample.depth is not None:
            depth_error = sample.depth.unsqueeze(0) - interface_depth
            violation = (
                (depth_error[..., 0] > 0.03)
                & (sample.depth.unsqueeze(0)[..., 0] > 0)
                & (transparent_mask[..., 0] > 0)
            )
            loss = loss + 0.5 * _weighted_mean(
                transparency[..., 0], violation.to(transparency.dtype)
            )

        output.trans_guidance_angle_weight = angle_weight
        output.trans_guidance_depth_weight = depth_weight
        output.trans_guidance_weight = guidance_weight
        output.trans_guidance_explainability_weight = explainability
        output.trans_guidance_depth_confidence = transparent_depth_confidence
        output.trans_guidance_depth_edge_weight = depth_edge_weight
        output.trans_guidance_coverage_weight = coverage_weight
        return loss

    def _multi_view_terms(
        self, output: Any, sample: Any, step: int
    ) -> tuple[Tensor, Tensor] | None:
        config = self.loss_config
        active = (
            config.multi_view > 0
            and step >= config.multi_view_start
            and (config.multi_view_stop is None or step < config.multi_view_stop)
            and step % config.multi_view_every == 0
        )
        if not active:
            return None
        candidates = (sample.previous, sample.next)
        neighbor = next(
            (
                frame
                for frame in candidates
                if frame is not None
                and (frame.camera_name, frame.frame_name)
                != (sample.camera_name, sample.frame_name)
            ),
            None,
        )
        if neighbor is None:
            return None

        pipe = AttrDict(compute_cov3D_python=False, depth_ratio=0.0)
        neighbor_render = render_glint_camera(
            neighbor.camera,
            self.model.interface,
            pipe,
            self.model.interface_background,
            device=str(self.device),
        )
        height, width = sample.camera.image_height, sample.camera.image_width
        depth = output.dpt_map.reshape(height, width, 1)
        neighbor_depth = neighbor_render.surf_depth.permute(1, 2, 0)
        rays = generate_camera_rays(sample.camera, z_depth=True)
        points = rays.origins + rays.directions * depth
        neighbor_pixels, neighbor_z = _project_world(points, neighbor.camera)
        sampled_neighbor_depth = _sample_image(neighbor_depth, neighbor_pixels)

        ones = torch.ones_like(sampled_neighbor_depth)
        neighbor_homogeneous = torch.cat((neighbor_pixels, ones), dim=-1)
        neighbor_camera_rays = torch.einsum(
            "ij,...j->...i", torch.linalg.inv(neighbor.camera.K), neighbor_homogeneous
        )
        neighbor_world_rays = torch.einsum(
            "ij,...j->...i", neighbor.camera.R.T, neighbor_camera_rays
        )
        neighbor_points = (
            neighbor.camera.camera_center.reshape(1, 1, 3)
            + neighbor_world_rays * sampled_neighbor_depth
        )
        roundtrip_pixels, roundtrip_z = _project_world(neighbor_points, sample.camera)
        y, x = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=depth.dtype) + 0.5,
            torch.arange(width, device=self.device, dtype=depth.dtype) + 0.5,
            indexing="ij",
        )
        reference_pixels = torch.stack((x, y), dim=-1)
        pixel_noise = (roundtrip_pixels - reference_pixels).norm(dim=-1)
        valid = (
            (depth[..., 0] > 0)
            & (neighbor_z[..., 0] > 0.1)
            & (roundtrip_z[..., 0] > 0.1)
            & (sampled_neighbor_depth[..., 0] > 0)
            & (neighbor_pixels[..., 0] >= 0)
            & (neighbor_pixels[..., 0] <= neighbor.camera.image_width - 1)
            & (neighbor_pixels[..., 1] >= 0)
            & (neighbor_pixels[..., 1] <= neighbor.camera.image_height - 1)
            & (pixel_noise < config.multi_view_pixel_noise)
        )
        if not valid.any():
            zero = depth.sum() * 0.0
            return zero, zero
        geometry_weight = torch.exp(-pixel_noise).detach()
        geometry = (geometry_weight[valid] * pixel_noise[valid]).mean()

        valid_indices = torch.where(valid.reshape(-1))[0]
        if len(valid_indices) > config.multi_view_samples:
            order = torch.randperm(len(valid_indices), device=self.device)[
                : config.multi_view_samples
            ]
            valid_indices = valid_indices[order]
        centers = reference_pixels.reshape(-1, 2)[valid_indices]
        centers_world = points.reshape(-1, 3)[valid_indices]
        normals = F.normalize(output.norm_map.reshape(-1, 3)[valid_indices], dim=-1)
        radius = config.multi_view_patch_size
        offsets_1d = torch.arange(
            -radius, radius + 1, device=self.device, dtype=depth.dtype
        )
        oy, ox = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
        offsets = torch.stack((ox, oy), dim=-1).reshape(1, -1, 2)
        current_patch_pixels = centers[:, None, :] + offsets
        homogeneous = torch.cat(
            (
                current_patch_pixels,
                torch.ones_like(current_patch_pixels[..., :1]),
            ),
            dim=-1,
        )
        camera_patch_rays = torch.einsum(
            "ij,...j->...i", torch.linalg.inv(sample.camera.K), homogeneous
        )
        world_patch_rays = torch.einsum(
            "ij,...j->...i", sample.camera.R.T, camera_patch_rays
        )
        camera_center = sample.camera.camera_center.reshape(1, 1, 3)
        numerator = (normals * (centers_world - camera_center[:, 0])).sum(
            dim=-1, keepdim=True
        )
        denominator = (
            (normals[:, None, :] * world_patch_rays)
            .sum(dim=-1)
            .clamp(min=-1e8, max=1e8)
        )
        stable = denominator.abs() > 1e-6
        distance = numerator / torch.where(
            stable, denominator, torch.ones_like(denominator)
        )
        patch_world = camera_center + world_patch_rays * distance[..., None]
        warped_pixels, warped_z = _project_world(patch_world, neighbor.camera)
        patch_valid = (
            stable
            & (distance > 0)
            & (warped_z[..., 0] > 0)
            & (warped_pixels[..., 0] >= 0)
            & (warped_pixels[..., 0] <= neighbor.camera.image_width - 1)
            & (warped_pixels[..., 1] >= 0)
            & (warped_pixels[..., 1] <= neighbor.camera.image_height - 1)
        ).all(dim=-1)
        reference_gray = (
            0.2989 * sample.rgb[..., 0]
            + 0.5870 * sample.rgb[..., 1]
            + 0.1140 * sample.rgb[..., 2]
        )[..., None]
        neighbor_gray = (
            0.2989 * neighbor.rgb[..., 0]
            + 0.5870 * neighbor.rgb[..., 1]
            + 0.1140 * neighbor.rgb[..., 2]
        )[..., None]
        reference_patch = _sample_image(reference_gray, current_patch_pixels)[..., 0]
        neighbor_patch = _sample_image(neighbor_gray, warped_pixels)[..., 0]
        ncc, ncc_valid = _local_ncc(reference_patch, neighbor_patch)
        ncc_valid = ncc_valid & patch_valid
        if ncc_valid.any():
            selected_weights = geometry_weight.reshape(-1)[valid_indices]
            ncc_loss = (ncc[ncc_valid] * selected_weights[ncc_valid]).mean()
        else:
            ncc_loss = geometry * 0.0
        return geometry, ncc_loss

    def _loss(
        self,
        output: Any,
        sample: Any,
        step: int,
    ) -> tuple[Tensor, dict[str, float]]:
        config = self.loss_config
        prediction = output.render.permute(1, 2, 0).unsqueeze(0)
        target = sample.rgb.unsqueeze(0)
        terms: dict[str, Tensor] = {}
        diagnostics: dict[str, Tensor | float] = {}
        terms["rgb_l1"] = F.l1_loss(prediction, target)
        terms["rgb_ssim"] = ssim_loss(
            prediction.permute(0, 3, 1, 2),
            target.permute(0, 3, 1, 2),
        )

        direct = getattr(output, "transmission_direct", None)
        direct_prediction = None
        if direct is not None:
            direct_prediction = direct.render.permute(1, 2, 0).unsqueeze(0)
            terms["transmission_rgb_l1"] = F.l1_loss(direct_prediction, target)
            terms["transmission_rgb_ssim"] = ssim_loss(
                direct_prediction.permute(0, 3, 1, 2),
                target.permute(0, 3, 1, 2),
            )

        normal_world = output.norm_map.reshape(
            1, sample.camera.image_height, sample.camera.image_width, 3
        )
        normal_camera = F.normalize(normal_world @ sample.camera.R.T, dim=-1)
        target_normal = (
            sample.stable_normal if config.normal_source == "stable" else sample.normal
        )
        if target_normal is not None:
            target_normal = F.normalize(target_normal.unsqueeze(0) * 2.0 - 1.0, dim=-1)
        normal_cosine = None
        normal_supervision_mask = None
        normal_threshold = config.normal_cosine_threshold(step)
        if target_normal is not None:
            normal_cosine, normal_supervision_mask = _normal_agreement(
                normal_camera,
                target_normal,
                normal_threshold,
            )
            if normal_threshold is not None:
                diagnostics["normal_cos_threshold"] = normal_threshold
                diagnostics[
                    "normal_supervision_valid_ratio"
                ] = normal_supervision_mask.float().mean()
        normal_prior_active = step >= config.normal_prior_start and (
            config.normal_prior_stop is None or step < config.normal_prior_stop
        )
        if target_normal is not None and normal_prior_active:
            assert normal_cosine is not None and normal_supervision_mask is not None
            normal_error = 1.0 - normal_cosine
            normal_weight = normal_supervision_mask.to(normal_error.dtype)
            if config.normal_depth_weight:
                depth = output.dpt_map.reshape(
                    1, sample.camera.image_height, sample.camera.image_width
                )
                normal_weight = normal_weight * _inverse_depth_weight(depth)
            terms["normal_prior"] = _weighted_mean(normal_error, normal_weight)

        depth_prior_active = step >= config.depth_prior_start and (
            config.depth_prior_stop is None or step < config.depth_prior_stop
        )
        if sample.depth is not None and depth_prior_active:
            rendered_depth = output.dpt_map.reshape(
                1, sample.camera.image_height, sample.camera.image_width, 1
            )
            depth_mask = None
            if (
                config.depth_normal_thresholding
                and normal_threshold is not None
                and normal_supervision_mask is not None
            ):
                depth_mask = normal_supervision_mask[..., None]
                diagnostics["depth_supervision_valid_ratio"] = (
                    (
                        depth_mask[..., 0]
                        & torch.isfinite(sample.depth[..., 0]).unsqueeze(0)
                        & (sample.depth[..., 0] > 0).unsqueeze(0)
                    )
                    .float()
                    .mean()
                )
            terms["depth_prior"] = _scale_shift_invariant_depth_loss(
                rendered_depth,
                sample.depth.unsqueeze(0),
                mask=depth_mask,
            )

        consistency_active = step >= config.normal_consistency_start and (
            config.normal_consistency_stop is None
            or step < config.normal_consistency_stop
        )
        if consistency_active:
            rendered_normal = output.interface.rend_normal.permute(1, 2, 0).unsqueeze(0)
            surface_normal = output.interface.surf_normal.permute(1, 2, 0).unsqueeze(0)
            consistency = 1.0 - F.cosine_similarity(
                rendered_normal, surface_normal, dim=-1
            )
            consistency_weight = torch.ones_like(consistency)
            if config.normal_consistency_depth_weight:
                consistency_weight = _inverse_depth_weight(
                    output.dpt_map.reshape(
                        1, sample.camera.image_height, sample.camera.image_width
                    )
                )
            terms["normal_consistency"] = _weighted_mean(
                consistency, consistency_weight
            )
            if direct is not None:
                direct_rendered_normal = direct.rend_normal.permute(1, 2, 0).unsqueeze(
                    0
                )
                direct_surface_normal = direct.surf_normal.permute(1, 2, 0).unsqueeze(0)
                direct_consistency = 1.0 - F.cosine_similarity(
                    direct_rendered_normal, direct_surface_normal, dim=-1
                )
                direct_weight = torch.ones_like(direct_consistency)
                if config.normal_consistency_depth_weight:
                    direct_weight = _inverse_depth_weight(
                        direct.surf_depth.permute(1, 2, 0).unsqueeze(0)[..., 0]
                    )
                terms["transmission_normal_consistency"] = _weighted_mean(
                    direct_consistency, direct_weight
                )

        if step >= config.normal_smooth_start:
            terms["normal_smooth"] = _edge_aware_smoothness(normal_world, target)

        material_transparency = getattr(output, "material_trans_map", output.trans_map)
        transparency = material_transparency.reshape(
            1, sample.camera.image_height, sample.camera.image_width, 1
        ).clamp(1e-6, 1.0 - 1e-6)
        guidance_active = step >= config.transparency_guidance_start and (
            config.transparency_guidance_stop is None
            or step < config.transparency_guidance_stop
        )
        if guidance_active:
            guidance = self._transparency_guidance(
                output, sample, transparency, target, step
            )
            if guidance is not None:
                terms["transparency_guidance"] = guidance

        if step >= config.transparency_regularization_start:
            (
                regularization,
                transparency_diagnostics,
            ) = _confidence_aware_transparency_regularization(
                transparency,
                target,
                getattr(output, "transparency_positive_mask", None),
                getattr(output, "transparency_negative_mask", None),
                config,
                step,
                primitive_transparency=self.model.interface.get_transmission_coeff,
                primitive_weight=(
                    output.interface.visibility_filter.reshape(-1, 1).to(
                        self.model.interface.get_opacity.dtype
                    )
                    * self.model.interface.get_opacity.detach()
                ),
            )
            terms["transparency_regularization"] = regularization
            diagnostics.update(transparency_diagnostics)

        plausibility_active = step >= config.plausibility_start and (
            config.plausibility_stop is None or step < config.plausibility_stop
        )
        if direct is not None and plausibility_active:
            opaque_mask = getattr(output, "confident_opaque_mask", None)
            if opaque_mask is not None:
                if sample.depth is not None:
                    opaque_mask = opaque_mask.bool() & (sample.depth.unsqueeze(0) > 0)
                direct_depth = direct.surf_depth.permute(1, 2, 0).unsqueeze(0)
                interface_depth = output.dpt_map.reshape(
                    1, sample.camera.image_height, sample.camera.image_width, 1
                ).detach()
                plausibility = _scale_shift_invariant_depth_loss(
                    direct_depth, interface_depth, opaque_mask
                )
                if target_normal is not None:
                    direct_normal_camera = F.normalize(
                        direct.rend_normal.permute(1, 2, 0).unsqueeze(0)
                        @ sample.camera.R.T,
                        dim=-1,
                    )
                    normal_error = 1.0 - (direct_normal_camera * target_normal).sum(
                        dim=-1
                    )
                    plausibility = plausibility + _weighted_mean(
                        normal_error, opaque_mask[..., 0]
                    )
                terms["plausibility"] = plausibility

        if (
            step >= config.transmission_opacity_start
            and (
                config.transmission_opacity_stop is None
                or step < config.transmission_opacity_stop
            )
            and output.transmission_trace is not None
        ):
            terms["transmission_opacity"] = (
                (1.0 - output.transmission_trace.alpha).abs().mean()
            )

        if (
            config.perceptual > 0
            and step > config.perceptual_start
            and step % config.perceptual_every == 0
        ):
            terms["perceptual"] = self._perceptual_loss(prediction, target)
            if direct_prediction is not None:
                terms["transmission_perceptual"] = self._perceptual_loss(
                    direct_prediction, target
                )

        multi_view = self._multi_view_terms(output, sample, step)
        if multi_view is not None:
            terms["multi_view_geo"] = multi_view[0]
            terms["multi_view_ncc"] = multi_view[1]

        weights = {
            "rgb_l1": config.rgb_l1,
            "rgb_ssim": config.rgb_ssim,
            "transmission_rgb_l1": config.rgb_l1,
            "transmission_rgb_ssim": config.rgb_ssim,
            "normal_prior": config.normal_prior,
            "depth_prior": config.depth_prior,
            "normal_consistency": config.normal_consistency,
            "transmission_normal_consistency": config.normal_consistency,
            "normal_smooth": config.normal_smooth,
            "transparency_regularization": config.transparency_regularization,
            "transparency_guidance": config.transparency_guidance,
            "transmission_opacity": config.transmission_opacity,
            "plausibility": config.plausibility,
            "perceptual": config.perceptual,
            "transmission_perceptual": config.perceptual,
            "multi_view_geo": config.multi_view * config.multi_view_geo,
            "multi_view_ncc": config.multi_view * config.multi_view_ncc,
        }
        total = sum(weights[name] * value for name, value in terms.items())
        scalars = {name: float(value.detach()) for name, value in terms.items()}
        scalars.update(
            {
                name: float(value.detach()) if torch.is_tensor(value) else float(value)
                for name, value in diagnostics.items()
            }
        )
        scalars["total"] = float(total.detach())
        return total, scalars

    def _save_training_image(self, output: Any, sample: Any, step: int) -> None:
        stem = f"step_{step:06d}_camera{sample.camera_name}_frame{sample.frame_name}"
        self.visualizer.save(
            output,
            sample,
            self.output_dir / "train_visualizations",
            stem,
        )

    def save(self, step: int, *, latest: bool = True) -> Path:
        optimizer_state = {
            set_name: {
                name: optimizer.state_dict() for name, optimizer in optimizers.items()
            }
            for set_name, optimizers in self.optimizers.items()
        }
        payload = {
            "model": self.model.legacy_state_dict(),
            "epoch": step,
            "step": step,
            "optimizers": optimizer_state,
            "refiners": {
                name: refiner.state_dict() for name, refiner in self.refiners.items()
            },
            "schedule": asdict(self.schedule),
        }
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"step_{step:06d}.pt"
        torch.save(payload, path)
        if latest:
            torch.save(payload, checkpoint_dir / "latest.pt")
        return path

    def restore_training_state(
        self,
        path: str | Path,
        *,
        legacy_steps_per_epoch: int = 1,
    ) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        for set_name, optimizer_states in payload.get("optimizers", {}).items():
            for name, state in optimizer_states.items():
                self.optimizers[set_name][name].load_state_dict(state)
        for name, state in payload.get("refiners", {}).items():
            self.refiners[name].load_state_dict(state)
        restored_covariance_stats = {
            name: refiner.stabilize_covariance(clear_boundary_momentum=True)
            for name, refiner in self.refiners.items()
        }
        if any(
            int(stats["scale_clamped"]) > 0 or int(stats["scale_momenta_cleared"]) > 0
            for stats in restored_covariance_stats.values()
        ):
            print(f"restored_covariance_stabilization={restored_covariance_stats}")
        if "step" in payload:
            self.step = int(payload["step"]) + 1
        else:
            # EasyVolCap stores a zero-based epoch rather than a global step.
            # Its monolithic optimizer state cannot be mapped safely after the
            # lean schema change, so continue at the right curriculum point
            # with freshly initialized per-parameter Adam states.
            self.step = (int(payload.get("epoch", -1)) + 1) * legacy_steps_per_epoch
            if "optimizer" in payload:
                print(
                    "legacy EasyVolCap optimizer state is not schema-compatible; "
                    f"resuming at global step {self.step} with fresh Adam states"
                )

    def _infinite_batches(self) -> Iterator[GlintBatch]:
        loader = make_glint_dataloader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.num_workers > 0,
            persistent_workers=self.config.num_workers > 0,
        )
        while True:
            yield from loader

    def train(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.model.train()
        batches = self._infinite_batches()
        last_stage: TrainingStage | None = None
        started = time.time()
        for step in range(self.step, self.config.max_steps):
            stage = self.schedule.stage(step)
            if stage != last_stage:
                print(f"[stage] step={step} -> {stage}")
                last_stage = stage
            batch = next(batches)
            sample = batch.samples[0].to(self.device)
            self._freeze_interface_geometry_if_needed(step)
            self._zero_grad()
            self._update_learning_rates(step)
            self._update_sh_degree(step)
            output = render_glint_transport(
                sample.camera,
                self.model.as_checkpoint(step),
                self.tracer if stage != "interface" else None,
                stage=stage,
                # Match released GLINT: transport losses may update interface
                # normals/materials, but ray-origin depth is detached.
                detach_surface_depth=True,
                render_transmission_direct=stage != "interface",
            )
            self.refiners["interface"].retain_interface_gradient(output)
            loss, scalars = self._loss(output, sample, step)
            loss.backward()

            active_sets = self._active_sets(step)
            # Apply the gradients to the parameters that produced this render
            # before topology refinement replaces/slices those parameters.
            # Refining first rebinds optimizer parameter groups and silently
            # drops the just-computed gradients on every refinement step.
            for name in active_sets:
                for parameter_name, optimizer in self.optimizers[name].items():
                    if (
                        name == "interface"
                        and self._interface_geometry_frozen
                        and parameter_name in _INTERFACE_GEOMETRY_PARAMETERS
                    ):
                        continue
                    optimizer.step()

            refinement_stats: dict[str, dict[str, int | float]] = {}
            if "interface" in active_sets:
                if self._interface_geometry_frozen:
                    refinement_stats["interface"] = self.refiners[
                        "interface"
                    ].stabilize_covariance()
                else:
                    self.refiners["interface"].accumulate_interface(output)
                    refinement_stats["interface"] = self.refiners["interface"].step(
                        step
                    )
            if "transmission" in active_sets:
                self.refiners["transmission"].accumulate_trace(
                    output.transmission_trace
                )
                refinement_stats["transmission"] = self.refiners["transmission"].step(
                    step
                )
            if "reflection" in active_sets:
                self.refiners["reflection"].accumulate_trace(output.reflection_trace)
                self.refiners["reflection"].accumulate_trace(
                    output.secondary_reflection_trace
                )
                refinement_stats["reflection"] = self.refiners["reflection"].step(step)

            if step % self.config.log_every == 0:
                counts = {
                    "interface": len(self.model.interface.get_xyz),
                    "transmission": len(self.model.transmission.get_xyz),
                    "reflection": len(self.model.reflection.get_xyz),
                }
                elapsed = time.time() - started
                print(
                    f"step={step:06d} stage={stage} loss={scalars['total']:.6f} "
                    f"sets={counts} elapsed={elapsed:.1f}s terms={scalars} "
                    f"refine={refinement_stats}"
                )
            if self.config.image_every > 0 and step % self.config.image_every == 0:
                self._save_training_image(output, sample, step)
            if (
                self.config.save_every > 0
                and step > 0
                and step % self.config.save_every == 0
            ):
                path = self.save(step)
                print(f"saved {path}")
            self.step = step + 1

        final_step = self.config.max_steps - 1
        path = self.save(final_step)
        print(f"training complete; checkpoint={path}")
