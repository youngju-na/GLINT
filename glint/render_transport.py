# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Render GLINT's complete decomposed Gaussian radiance transport pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from glint.camera import load_easyvolcap_camera
from glint.checkpoint import load_glint_checkpoint
from glint.optix_backend import OptixSurfelTracer
from glint.torch_tracer import TorchSurfelTracer
from glint.transport import render_glint_transport


def _write_image(path: Path, image: torch.Tensor) -> None:
    image = image.detach().permute(1, 2, 0).clamp(0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, (image.cpu().numpy() * 255.0).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--camera", default="0000")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components-dir", type=Path)
    parser.add_argument("--backend", choices=("optix", "torch"), default="optix")
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow pickle loading for a trusted legacy EasyVolCap checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    checkpoint = load_glint_checkpoint(
        args.checkpoint,
        device=device,
        trust_checkpoint=args.trust_checkpoint,
    )
    camera = load_easyvolcap_camera(
        args.data_root, args.camera, ratio=args.ratio, device=device
    )
    tracer = OptixSurfelTracer() if args.backend == "optix" else TorchSurfelTracer()
    tracer.eval()
    with torch.no_grad():
        output = render_glint_transport(
            camera,
            checkpoint,
            tracer,
        )

    _write_image(args.output, output.render)
    if args.components_dir is not None:
        for name, image in (
            ("interface", output.interface.render),
            ("diffuse_weighted", output.dif_render),
            ("reflection_weighted", output.ref_render),
            ("transmission_weighted", output.trans_render),
        ):
            _write_image(args.components_dir / f"{name}.png", image)

    counts = ", ".join(
        f"{name}={getattr(checkpoint, name).get_xyz.shape[0]}"
        for name in ("pcd", "env", "trans_env")
    )
    energy_error = (output.energy_sum - 1.0).abs().max().item()
    print(
        f"rendered camera={args.camera} backend={args.backend} {counts} "
        f"max_energy_error={energy_error:.3e} output={args.output}"
    )


if __name__ == "__main__":
    main()
