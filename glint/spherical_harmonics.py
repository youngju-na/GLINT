# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Small PyTorch spherical-harmonics evaluator for the reference tracer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def evaluate_spherical_harmonics(degree: int, directions: Tensor, coeffs: Tensor) -> Tensor:
    """Evaluate real SH coefficients up to degree three without a CUDA extension."""

    if degree < 0 or degree > 3:
        raise ValueError(f"GLINT supports SH degrees 0 through 3, got {degree}")
    basis_count = (degree + 1) ** 2
    if directions.shape[-1] != 3:
        raise ValueError(f"directions must end in 3 values, got {directions.shape}")
    if coeffs.ndim < 2 or coeffs.shape[-1] != 3:
        raise ValueError(f"coefficients must end in [K, 3], got {coeffs.shape}")
    if coeffs.shape[-2] < basis_count:
        raise ValueError(f"degree {degree} requires {basis_count} coefficients")

    try:
        batch_shape = torch.broadcast_shapes(directions.shape[:-1], coeffs.shape[:-2])
    except RuntimeError as exc:
        raise ValueError(
            f"directions {directions.shape} and coefficients {coeffs.shape} "
            "are not batch-broadcastable"
        ) from exc
    directions = F.normalize(directions.expand(*batch_shape, 3), dim=-1)
    coeffs = coeffs.expand(*batch_shape, coeffs.shape[-2], 3)
    x, y, z = directions.unbind(dim=-1)
    basis = torch.empty(
        (*directions.shape[:-1], basis_count),
        device=directions.device,
        dtype=directions.dtype,
    )
    basis[..., 0] = 0.2820947917738781
    if degree >= 1:
        basis[..., 1] = -0.48860251190292 * y
        basis[..., 2] = 0.48860251190292 * z
        basis[..., 3] = -0.48860251190292 * x
    if degree >= 2:
        z2 = z.square()
        xy2 = 2.0 * x * y
        x2_minus_y2 = x.square() - y.square()
        basis[..., 4] = 0.5462742152960395 * xy2
        basis[..., 5] = -1.092548430592079 * z * y
        basis[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
        basis[..., 7] = -1.092548430592079 * z * x
        basis[..., 8] = 0.5462742152960395 * x2_minus_y2
    if degree >= 3:
        z2 = z.square()
        xy2 = 2.0 * x * y
        x2_minus_y2 = x.square() - y.square()
        harmonic_cos2 = x * x2_minus_y2 - y * xy2
        harmonic_sin2 = x * xy2 + y * x2_minus_y2
        common = -2.285228997322329 * z2 + 0.4570457994644658
        basis[..., 9] = -0.5900435899266435 * harmonic_sin2
        basis[..., 10] = 1.445305721320277 * z * xy2
        basis[..., 11] = common * y
        basis[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
        basis[..., 13] = common * x
        basis[..., 14] = 1.445305721320277 * z * x2_minus_y2
        basis[..., 15] = -0.5900435899266435 * harmonic_cos2
    return (basis[..., None] * coeffs[..., :basis_count, :]).sum(dim=-2)
