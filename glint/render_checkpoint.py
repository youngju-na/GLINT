# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Render GLINT's interaction Gaussian set from an existing checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import torch

from glint.camera import load_easyvolcap_camera
from glint.checkpoint import load_glint_checkpoint
from glint.renderer import render_glint_camera


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--camera", default="0000")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow pickle loading for a trusted legacy EasyVolCap checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    checkpoint = load_glint_checkpoint(
        args.checkpoint,
        gaussian_sets=("pcd",),
        device=device,
        trust_checkpoint=args.trust_checkpoint,
    )
    camera = load_easyvolcap_camera(
        args.data_root, args.camera, ratio=args.ratio, device=device
    )
    pipe = SimpleNamespace(compute_cov3D_python=False, depth_ratio=0.0)
    with torch.no_grad():
        output = render_glint_camera(camera, checkpoint.pcd, pipe, checkpoint.bg_color)

    image = output.render.permute(1, 2, 0).clamp(0.0, 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output, (image.cpu().numpy() * 255.0).astype(np.uint8))
    print(
        f"rendered camera={args.camera} gaussians={checkpoint.pcd.get_xyz.shape[0]} "
        f"visible={int(output.visibility_filter.sum())} output={args.output}"
    )


if __name__ == "__main__":
    main()
