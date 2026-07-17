# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""2D-surface densification and pruning for the standalone GLINT trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.cuda._math import _quat_to_rotmat
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove

from glint.checkpoint import GlintGaussianSet


@dataclass(frozen=True)
class RefinementConfig:
    start: int = 500
    stop: int = 21_000
    every: int = 100
    middle_start: int | None = None
    middle_stop: int | None = None
    middle_every: int | None = None
    reset_every: int = 3_000
    grow_gradient: float = 2e-4
    grow_scale: float = 0.01
    prune_scale: float = 0.1
    prune_opacity: float = 0.05
    max_gaussians: int = 1_000_000
    min_gaussians: int = 1
    min_gaussian_ratio: float = 0.0
    adaptive_grow_quantile: float | None = None
    large_gaussian_weight_quantile: float | None = None
    max_scale_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < 0:
            raise ValueError("refinement start/stop must be non-negative")
        if self.every <= 0 or self.reset_every <= 0:
            raise ValueError("refinement intervals must be positive")
        if self.middle_every is not None and self.middle_every <= 0:
            raise ValueError("middle refinement interval must be positive")
        if self.max_gaussians <= 0:
            raise ValueError("max_gaussians must be positive")
        if self.min_gaussians <= 0:
            raise ValueError("min_gaussians must be positive")
        if self.min_gaussians > self.max_gaussians:
            raise ValueError("min_gaussians must not exceed max_gaussians")
        if not 0.0 <= self.min_gaussian_ratio <= 1.0:
            raise ValueError("min_gaussian_ratio must be in [0, 1]")
        for name, value in (
            ("adaptive_grow_quantile", self.adaptive_grow_quantile),
            ("large_gaussian_weight_quantile", self.large_gaussian_weight_quantile),
        ):
            if value is not None and not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1) or None")
        if self.max_scale_ratio is not None and self.max_scale_ratio <= 0:
            raise ValueError("max_scale_ratio must be positive or None")


def _split_2d(
    parameters: dict[str, torch.nn.Parameter],
    optimizers: dict[str, torch.optim.Optimizer],
    mask: Tensor,
    state: dict[str, Tensor] | None = None,
) -> None:
    """Split surfels in their two-dimensional tangent plane."""

    selected = torch.where(mask)[0]
    rest = torch.where(~mask)[0]
    scales = torch.exp(parameters["scales"][selected])
    rotations = _quat_to_rotmat(F.normalize(parameters["quats"][selected], dim=-1))
    local_samples = torch.zeros(
        2, len(selected), 3, device=mask.device, dtype=scales.dtype
    )
    local_samples[..., :2] = (
        torch.randn(2, len(selected), 2, device=mask.device) * scales[None]
    )
    samples = torch.einsum("nij,bnj->bni", rotations, local_samples)

    def parameter_fn(name: str, parameter: Tensor) -> Tensor:
        repeats = [2] + [1] * (parameter.ndim - 1)
        if name == "means":
            children = (parameter[selected] + samples).reshape(-1, 3)
        elif name == "scales":
            children = torch.log(scales / 1.6).repeat(2, 1)
        else:
            children = parameter[selected].repeat(repeats)
        return torch.nn.Parameter(
            torch.cat((parameter[rest], children)),
            requires_grad=parameter.requires_grad,
        )

    def optimizer_fn(_key: str, value: Tensor) -> Tensor:
        children = torch.zeros(
            (2 * len(selected), *value.shape[1:]),
            device=value.device,
            dtype=value.dtype,
        )
        return torch.cat((value[rest], children))

    _update_param_with_optimizer(
        parameter_fn,
        optimizer_fn,
        parameters,
        optimizers,
    )
    if state is not None:
        for name, value in state.items():
            repeats = [2] + [1] * (value.ndim - 1)
            state[name] = torch.cat((value[rest], value[selected].repeat(repeats)))


class GlintRefiner:
    """Accumulate visibility gradients and mutate one Gaussian set safely."""

    def __init__(
        self,
        gaussian_set: GlintGaussianSet,
        optimizers: dict[str, torch.optim.Optimizer],
        config: RefinementConfig,
        *,
        scene_scale: float,
    ) -> None:
        if not math.isfinite(scene_scale) or scene_scale <= 0:
            raise ValueError("scene_scale must be finite and positive")
        self.gaussian_set = gaussian_set
        self.optimizers = optimizers
        self.config = config
        self.scene_scale = float(scene_scale)
        self.initial_count = len(gaussian_set.get_xyz)
        self.grad_accum: Tensor | None = None
        self.count: Tensor | None = None
        self.weight_accum: Tensor | None = None

    def _ensure_state(self) -> None:
        count = len(self.gaussian_set.get_xyz)
        device = self.gaussian_set.get_xyz.device
        if (
            self.grad_accum is None
            or self.count is None
            or len(self.grad_accum) != count
        ):
            self.grad_accum = torch.zeros(count, device=device)
            self.count = torch.zeros(count, device=device)
        if self.weight_accum is None or len(self.weight_accum) != count:
            self.weight_accum = torch.zeros(count, device=device)

    def retain_interface_gradient(self, output: Any) -> None:
        gradient = output.interface.native_meta["gradient_2dgs"]
        if gradient.requires_grad:
            gradient.retain_grad()

    @torch.no_grad()
    def accumulate_interface(self, output: Any) -> None:
        self._ensure_state()
        meta = output.interface.native_meta
        gradient = meta["gradient_2dgs"].grad
        if gradient is None:
            return
        gradient = gradient[0] if gradient.ndim == 3 else gradient
        gradient = gradient.clone()
        gradient[..., 0] *= meta["width"] / 2.0
        gradient[..., 1] *= meta["height"] / 2.0
        radii = meta["radii"][0].amax(dim=-1)
        visible = radii > 0
        assert self.grad_accum is not None and self.count is not None
        self.grad_accum[visible] += gradient[visible].norm(dim=-1)
        self.count[visible] += 1

    @torch.no_grad()
    def accumulate_trace(self, trace_result: Any | None) -> None:
        if trace_result is None or trace_result.backend_data is None:
            return
        self._ensure_state()
        proxy = trace_result.backend_data.get("viewspace_points")
        if proxy is None or proxy.grad is None:
            return
        gradient = proxy.grad
        visible = gradient.norm(dim=-1) > 0
        weights = None
        if trace_result.weight_accumulate is not None:
            weights = trace_result.weight_accumulate.reshape(-1)
            if len(weights) == len(visible):
                visible |= weights > 0
            else:
                weights = None
        assert (
            self.grad_accum is not None
            and self.count is not None
            and self.weight_accum is not None
        )
        self.grad_accum[visible] += gradient[visible].norm(dim=-1)
        self.count[visible] += 1
        if weights is not None:
            self.weight_accum[visible] += weights[visible]

    def _limit_mask(self, mask: Tensor, scores: Tensor, capacity: int) -> Tensor:
        selected = torch.where(mask)[0]
        if len(selected) <= capacity:
            return mask
        keep = selected[torch.topk(scores[selected], capacity).indices]
        limited = torch.zeros_like(mask)
        limited[keep] = True
        return limited

    def _interval(self, step: int) -> int:
        config = self.config
        if (
            config.middle_every is not None
            and config.middle_start is not None
            and step >= config.middle_start
            and (config.middle_stop is None or step < config.middle_stop)
        ):
            return config.middle_every
        return config.every

    @torch.no_grad()
    def stabilize_covariance(
        self,
        *,
        clear_boundary_momentum: bool = False,
    ) -> dict[str, int | float]:
        """Project tangent covariance eigenvalues into a scene-relative bound.

        A 2D Gaussian's nonzero covariance eigenvalues are the squares of its
        two tangent scales.  Bounding the log-scales therefore directly bounds
        the maximum covariance eigenvalue without depending on camera
        resolution or a particular scene's world units.  Projected Adam
        moments are cleared so momentum cannot repeatedly drive a component
        back outside the feasible set.
        """

        ratio = self.config.max_scale_ratio
        default_stats: dict[str, int | float] = {
            "scale_clamped": 0,
            "scale_components_clamped": 0,
            "scale_momenta_cleared": 0,
            "max_tangent_scale_ratio": 0.0,
            "max_tangent_covariance_eigenvalue_ratio": 0.0,
        }
        if ratio is None:
            return default_stats
        parameter = self.gaussian_set.parameter_map()["scales"]
        maximum = math.log(max(self.scene_scale * ratio, 1e-8))
        # Keep exp(log_scale) finite as well.  The lower bound is deliberately
        # very loose: it guards numerical corruption rather than discouraging
        # legitimately small surfels.
        minimum = math.log(max(self.scene_scale * 1e-8, 1e-12))
        data = parameter.data
        finite = torch.isfinite(data)
        projected = (~finite) | (data > maximum) | (data < minimum)
        if projected.any():
            if (~finite).any():
                finite_values = data[finite]
                fallback = (
                    finite_values.median().clamp(min=minimum, max=maximum)
                    if finite_values.numel()
                    else data.new_tensor(maximum)
                )
                data.nan_to_num_(
                    nan=float(fallback.item()),
                    posinf=maximum,
                    neginf=minimum,
                )
            data.clamp_(min=minimum, max=maximum)
        momentum_mask = projected.clone()
        if clear_boundary_momentum:
            # A resumed parameter may already have been projected before its
            # Adam state was restored.  Clear components sitting at the hard
            # boundary so stale checkpoint momentum cannot immediately push
            # them out again.
            momentum_mask |= data >= maximum - 1e-7
        if momentum_mask.any():
            optimizer = self.optimizers["scales"]
            state = optimizer.state.get(parameter, {})
            cleared_momenta = False
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state:
                    state[key][momentum_mask] = 0
                    cleared_momenta = True
        else:
            cleared_momenta = False

        maximum_log_scale = float(data.max().item()) if data.numel() else minimum
        maximum_scale_ratio = math.exp(maximum_log_scale) / self.scene_scale
        return {
            "scale_clamped": int(projected.any(dim=-1).sum().item()),
            "scale_components_clamped": int(projected.sum().item()),
            "scale_momenta_cleared": (
                int(momentum_mask.sum().item()) if cleared_momenta else 0
            ),
            "max_tangent_scale_ratio": maximum_scale_ratio,
            "max_tangent_covariance_eigenvalue_ratio": maximum_scale_ratio**2,
        }

    @torch.no_grad()
    def _reset_parameter(self, name: str, value: float) -> None:
        """Cap one sigmoid parameter and clear the matching Adam momentum.

        GLINT replaces the opacity tensor during a reset, which also zeros its
        first- and second-moment buffers.  An in-place value copy leaves stale
        Adam momentum attached and caused the weak reflection set to be driven
        below the pruning threshold after every reset.
        """

        parameters = self.gaussian_set.parameter_map()
        old_parameter = parameters[name]
        new_value = torch.minimum(
            old_parameter.detach(), old_parameter.new_tensor(value).logit()
        )
        new_parameter = torch.nn.Parameter(
            new_value, requires_grad=old_parameter.requires_grad
        )
        optimizer = self.optimizers[name]
        state = optimizer.state.pop(old_parameter, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state:
                state[key] = torch.zeros_like(new_parameter)
        for group in optimizer.param_groups:
            group["params"] = [new_parameter]
        if state:
            optimizer.state[new_parameter] = state
        parameters[name] = new_parameter
        self.gaussian_set.replace_parameter_map(parameters)

    @torch.no_grad()
    def step(self, step: int) -> dict[str, int | float]:
        config = self.config
        covariance_stats = self.stabilize_covariance()
        stats: dict[str, int | float] = {
            "duplicated": 0,
            "split": 0,
            "pruned": 0,
            **covariance_stats,
        }
        if step <= config.start or step >= config.stop:
            return stats
        self._ensure_state()
        assert (
            self.grad_accum is not None
            and self.count is not None
            and self.weight_accum is not None
        )

        if step % self._interval(step) == 0:
            parameters = self.gaussian_set.parameter_map()
            gradients = self.grad_accum / self.count.clamp_min(1)
            observed = self.count > 0
            positive_gradients = gradients[observed & torch.isfinite(gradients)]
            grow_threshold = config.grow_gradient
            if config.adaptive_grow_quantile is not None and positive_gradients.numel():
                adaptive = torch.quantile(
                    positive_gradients,
                    config.adaptive_grow_quantile,
                ).item()
                if adaptive > 0:
                    grow_threshold = min(grow_threshold, adaptive)
            average_weight = self.weight_accum / self.count.clamp_min(1)
            topology_state = {
                "average_weight": average_weight,
                "observed": observed.to(average_weight.dtype),
            }
            sizes = torch.exp(parameters["scales"]).amax(dim=-1)
            high_gradient = observed & (gradients >= grow_threshold)
            small = sizes <= config.grow_scale * self.scene_scale
            duplicate_mask = high_gradient & small
            split_mask = high_gradient & ~small

            current = len(sizes)
            capacity = max(0, config.max_gaussians - current)
            duplicate_mask = self._limit_mask(duplicate_mask, gradients, capacity)
            duplicate_count = int(duplicate_mask.sum().item())
            capacity -= duplicate_count
            # A split replaces one parent by two children, so it consumes one
            # additional slot rather than two.
            split_mask = self._limit_mask(split_mask, gradients, capacity)
            split_count = int(split_mask.sum().item())

            if duplicate_count:
                duplicate(
                    parameters,
                    self.optimizers,
                    topology_state,
                    duplicate_mask,
                )
                split_mask = torch.cat(
                    (
                        split_mask,
                        torch.zeros(
                            duplicate_count,
                            dtype=torch.bool,
                            device=split_mask.device,
                        ),
                    )
                )
            if split_count:
                _split_2d(
                    parameters,
                    self.optimizers,
                    split_mask,
                    topology_state,
                )

            opacities = torch.sigmoid(parameters["opacities"].flatten())
            sizes = torch.exp(parameters["scales"]).amax(dim=-1)
            prune_mask = opacities < config.prune_opacity
            if step > config.reset_every:
                oversized = sizes > config.prune_scale * self.scene_scale
                quantile = config.large_gaussian_weight_quantile
                if quantile is None:
                    prune_mask |= oversized
                else:
                    current_observed = topology_state["observed"] > 0
                    weights = topology_state["average_weight"]
                    if current_observed.any():
                        cutoff = torch.quantile(weights[current_observed], quantile)
                        low_weight = (~current_observed) | (weights <= cutoff)
                        prune_mask |= oversized & low_weight
                    else:
                        prune_mask |= oversized
            # A role must retain enough support to recover from an opacity
            # reset.  Keeping only one surfel made the reflection branch
            # irreversibly collapse in real training.
            ratio_floor = math.ceil(self.initial_count * config.min_gaussian_ratio)
            minimum = min(
                config.max_gaussians,
                max(config.min_gaussians, ratio_floor),
            )
            max_prunable = max(0, len(opacities) - minimum)
            if int(prune_mask.sum().item()) > max_prunable:
                opacity_score = (config.prune_opacity - opacities).clamp_min(0.0)
                scale_score = (
                    sizes / (config.prune_scale * self.scene_scale) - 1.0
                ).clamp_min(0.0)
                prune_mask = self._limit_mask(
                    prune_mask, opacity_score + scale_score, max_prunable
                )
            prune_count = int(prune_mask.sum().item())
            if prune_count:
                remove(parameters, self.optimizers, {}, prune_mask)

            self.gaussian_set.replace_parameter_map(parameters)
            stats = {
                "duplicated": duplicate_count,
                "split": split_count,
                "pruned": prune_count,
                **covariance_stats,
                "grow_threshold": float(grow_threshold),
                "observed": int(observed.sum().item()),
                "minimum": int(minimum),
            }
            self.grad_accum = None
            self.count = None
            self.weight_accum = None

        if step > 0 and step % config.reset_every == 0:
            self._reset_parameter("opacities", 0.01)
            if self.gaussian_set.role == "interface":
                self._reset_parameter("transparency", 0.01)
        return stats

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {
            "grad_accum": self.grad_accum,
            "count": self.count,
            "weight_accum": self.weight_accum,
            "initial_count": self.initial_count,
        }

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        self.grad_accum = state.get("grad_accum")
        self.count = state.get("count")
        self.weight_accum = state.get("weight_accum")
        initial_count = state.get("initial_count")
        if isinstance(initial_count, int):
            self.initial_count = initial_count
