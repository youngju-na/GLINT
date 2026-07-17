# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Small geometry operations shared by GLINT backends."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def quaternion_to_rotation_matrix(quaternions: Tensor) -> Tensor:
    """Convert scalar-first quaternions ``[..., 4]`` to matrices ``[..., 3, 3]``."""

    quaternions = F.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = torch.unbind(quaternions, dim=-1)
    matrix = torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))
