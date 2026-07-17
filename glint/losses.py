# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Photometric losses used by the GLINT trainer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


_WINDOW_CACHE: dict[tuple[int, int, torch.device, torch.dtype], Tensor] = {}


def _ssim_window(image: Tensor, window_size: int) -> Tensor:
    key = (window_size, image.shape[1], image.device, image.dtype)
    cached = _WINDOW_CACHE.get(key)
    if cached is not None:
        return cached

    coordinate = torch.arange(window_size, device=image.device, dtype=image.dtype)
    coordinate = coordinate - window_size // 2
    kernel = torch.exp(-(coordinate.square()) / (2.0 * 1.5**2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel)
    window = window.expand(image.shape[1], 1, window_size, window_size).contiguous()
    _WINDOW_CACHE[key] = window
    return window


def ssim_loss(prediction: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    """Return ``1 - SSIM`` for two ``[B, C, H, W]`` image batches."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            "SSIM expects equally shaped [B, C, H, W] tensors, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    window = _ssim_window(prediction, window_size)
    padding = window_size // 2
    channels = prediction.shape[1]
    mean_prediction = F.conv2d(
        prediction, window, padding=padding, groups=channels
    )
    mean_target = F.conv2d(target, window, padding=padding, groups=channels)
    prediction_variance = F.conv2d(
        prediction.square(), window, padding=padding, groups=channels
    ) - mean_prediction.square()
    target_variance = F.conv2d(
        target.square(), window, padding=padding, groups=channels
    ) - mean_target.square()
    covariance = F.conv2d(
        prediction * target, window, padding=padding, groups=channels
    ) - mean_prediction * mean_target
    c1 = 0.01**2
    c2 = 0.03**2
    similarity = (
        (2.0 * mean_prediction * mean_target + c1) * (2.0 * covariance + c2)
    ) / (
        (mean_prediction.square() + mean_target.square() + c1)
        * (prediction_variance + target_variance + c2)
    )
    return 1.0 - similarity.mean()
