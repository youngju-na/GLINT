# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Render a GLINT dataset split and compare it with ground-truth RGB images."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from glint.checkpoint import load_glint_checkpoint
from glint.dataset import (
    GlintDataset,
    load_easyvolcap_config,
    make_glint_dataloader,
)
from glint.optix_backend import OptixSurfelTracer
from glint.torch_tracer import TorchSurfelTracer
from glint.transport import render_glint_transport
from glint.visualizer import (
    DEFAULT_VISUALIZATION_TYPES,
    GlintVisualizationConfig,
    GlintVisualizer,
    include_diagnostic_visualizations,
)


def _write_image(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().clamp(0.0, 1.0).cpu().numpy()
    iio.imwrite(path, (image * 255.0).round().astype(np.uint8))


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    mse = error.square().mean().item()
    return {
        "mae": error.abs().mean().item(),
        "mse": mse,
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
    }


def _save_geometry_maps(
    output_dir: Path,
    output: Any,
    sample: Any,
) -> None:
    """Save z-depth and camera-space normals for geometry evaluation."""

    height = int(sample.camera.image_height)
    width = int(sample.camera.image_width)
    depth = output.dpt_map.reshape(height, width)
    normal_world = output.norm_map.reshape(height, width, 3)
    normal_world = F.normalize(normal_world, dim=-1)
    normal_camera = normal_world @ sample.camera.R.T
    depth_directory = output_dir / "geometry" / "depths_pred"
    normal_directory = output_dir / "geometry" / "normals_pred"
    depth_directory.mkdir(parents=True, exist_ok=True)
    normal_directory.mkdir(parents=True, exist_ok=True)
    np.save(
        depth_directory / f"{sample.camera_name}.npy",
        depth.detach().cpu().numpy().astype(np.float32),
    )
    np.save(
        normal_directory / f"{sample.camera_name}.npy",
        normal_camera.detach().cpu().numpy().astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        help="GLINT dataset or experiment YAML; inherited configs are resolved.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Dataset root, or an override for paths stored in --config.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-views", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backend", choices=("optix", "torch"), default="optix")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument(
        "--save-geometry",
        action="store_true",
        help="Save z-depth and camera-space normal arrays for geometry evaluation.",
    )
    parser.add_argument("--visualization-types", nargs="+")
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow pickle loading for a trusted legacy EasyVolCap checkpoint.",
    )
    args = parser.parse_args()
    if args.config is None and args.data_root is None:
        parser.error("one of --config or --data-root is required")

    dataset_overrides = {"load_sky_mask": False, "load_neighbors": False}
    if not args.save_visualizations or args.config is None:
        dataset_overrides.update({"load_stable_normal": False, "diffren_params": ()})
    if args.ratio is not None:
        dataset_overrides["ratio"] = args.ratio
    if args.data_root is not None and args.config is not None:
        dataset_overrides["data_root"] = args.data_root
    if args.config is not None:
        dataset = GlintDataset.from_config(
            args.config,
            split=args.split,
            **dataset_overrides,
        )
    else:
        dataset = GlintDataset(
            args.data_root,
            ratio=0.5 if args.ratio is None else args.ratio,
            split=args.split,
            **dataset_overrides,
        )

    if not 0 <= args.start_index < len(dataset):
        raise ValueError(
            f"start-index {args.start_index} is outside dataset length {len(dataset)}"
        )
    stop = len(dataset)
    if args.max_views is not None:
        if args.max_views <= 0:
            raise ValueError("max-views must be positive")
        stop = min(stop, args.start_index + args.max_views)

    device = torch.device("cuda")
    checkpoint = load_glint_checkpoint(
        args.checkpoint,
        device=device,
        trust_checkpoint=args.trust_checkpoint,
    )
    tracer = OptixSurfelTracer() if args.backend == "optix" else TorchSurfelTracer()
    tracer.eval()
    experiment_config = (
        load_easyvolcap_config(args.config) if args.config is not None else {}
    )
    runner = experiment_config.get("runner_cfg", {})
    supervisor = experiment_config.get("model_cfg", {}).get("supervisor_cfg", {})
    configured_visualization = runner.get("visualizer_cfg", {}).get(
        "types", DEFAULT_VISUALIZATION_TYPES
    )
    visualization_types = (
        tuple(str(name).upper() for name in args.visualization_types)
        if args.visualization_types is not None
        else include_diagnostic_visualizations(configured_visualization)
    )
    visualizer = GlintVisualizer(
        GlintVisualizationConfig(
            types=visualization_types,
            normal_source=str(supervisor.get("use_normal_type", "diffren")),
        )
    )
    loader = make_glint_dataloader(
        torch.utils.data.Subset(dataset, range(args.start_index, stop)),  # type: ignore[arg-type]
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0,
    )

    rows: list[dict[str, float | int | str]] = []
    saved_geometry_views: set[str] = set()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch in loader:
            sample = batch.samples[0].to(device)
            output = render_glint_transport(
                sample.camera,
                checkpoint,
                tracer,
                render_transmission_direct=args.save_visualizations,
            )
            prediction = output.render.permute(1, 2, 0).clamp(0.0, 1.0)
            target = sample.rgb
            metrics = _metrics(prediction, target)
            row: dict[str, float | int | str] = {
                "index": sample.index,
                "camera": sample.camera_name,
                "frame": sample.frame_name,
                **metrics,
            }
            rows.append(row)
            print(
                f"[{len(rows):04d}/{stop - args.start_index:04d}] "
                f"camera={sample.camera_name} PSNR={metrics['psnr']:.3f} "
                f"MAE={metrics['mae']:.5f}"
            )
            stem = f"{sample.index:04d}_{sample.camera_name}_{sample.frame_name}"
            if args.save_images:
                _write_image(args.output_dir / "pred" / f"{stem}.png", prediction)
                _write_image(args.output_dir / "gt" / f"{stem}.png", target)
                comparison = torch.cat((target, prediction), dim=1)
                _write_image(args.output_dir / "comparison" / f"{stem}.png", comparison)
            if args.save_visualizations:
                visualizer.save(
                    output,
                    sample,
                    args.output_dir / "visualizations",
                    stem,
                )
            if args.save_geometry:
                if sample.camera_name in saved_geometry_views:
                    raise ValueError(
                        "geometry evaluation expects one frame per camera; "
                        f"camera {sample.camera_name} occurs more than once"
                    )
                _save_geometry_maps(args.output_dir, output, sample)
                saved_geometry_views.add(sample.camera_name)

    aggregate = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in ("mae", "mse", "psnr")
    }
    report = {
        "dataset": dataset.summary(),
        "checkpoint": str(args.checkpoint.resolve()),
        "backend": args.backend,
        "evaluated_samples": len(rows),
        "aggregate": aggregate,
        "samples": rows,
    }
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"mean PSNR={aggregate['psnr']:.3f} MAE={aggregate['mae']:.5f} "
        f"over {len(rows)} samples; report={report_path}"
    )


if __name__ == "__main__":
    main()
