# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Evaluate GLINT depth, normal, and reconstructed geometry."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from glint.geometry_metrics import (
    depth_metrics,
    infer_map_shape,
    normal_metrics,
    point_cloud_metrics,
    resize_depth_target,
    resize_normal_target,
    sample_mesh,
)


DEFAULT_VIEW_SPLIT = (
    Path(__file__).resolve().parents[1] / "configs/evaluation/3d-front-t.json"
)


def _frame_index(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"prediction filename has no numeric frame index: {path.name}")
    return int(match.group(1))


def _prediction_map(
    value: np.ndarray, shape: tuple[int, int], channels: int
) -> np.ndarray:
    value = np.asarray(value).squeeze()
    expected_shape = shape if channels == 1 else (*shape, channels)
    if value.shape == expected_shape:
        return value.astype(np.float32, copy=False)
    if value.size != int(np.prod(expected_shape)):
        raise ValueError(f"cannot reshape prediction {value.shape} to {expected_shape}")
    return value.reshape(expected_shape).astype(np.float32, copy=False)


def _mean_metrics(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _select_depth_files(
    args: argparse.Namespace, depth_directory: Path
) -> tuple[list[Path], str, list[int] | None]:
    depth_files = sorted(depth_directory.glob("*.npy"))
    scene = args.scene or args.ground_truth_root.name
    if args.all_views:
        return depth_files, scene, None
    split = json.loads(args.view_split.read_text())
    if scene not in split:
        available = ", ".join(sorted(split))
        raise KeyError(
            f"scene {scene!r} is not in {args.view_split}; available scenes: {available}. "
            "Use --all-views for a custom dataset."
        )
    selected_frames = [int(frame) for frame in split[scene]]
    indexed = {_frame_index(path): path for path in depth_files}
    missing = [frame for frame in selected_frames if frame not in indexed]
    if missing and not args.skip_missing:
        raise FileNotFoundError(
            "missing predictions for official evaluation view(s): "
            + ", ".join(f"{frame:04d}" for frame in missing)
        )
    return (
        [indexed[frame] for frame in selected_frames if frame in indexed],
        scene,
        selected_frames,
    )


def evaluate_maps(args: argparse.Namespace) -> dict[str, object]:
    depth_directory = args.prediction_root / "depths_pred"
    normal_directory = args.prediction_root / "normals_pred"
    depth_files, scene, selected_frames = _select_depth_files(args, depth_directory)
    if not depth_files:
        raise FileNotFoundError(f"no depth predictions found under {depth_directory}")

    rows: list[dict[str, object]] = []
    for depth_path in depth_files:
        frame = _frame_index(depth_path)
        normal_path = normal_directory / f"{depth_path.stem}.npy"
        depth_target_path = (
            args.ground_truth_root / "depths_gt" / f"val_depthZ_{frame:04d}.npy"
        )
        normal_target_path = (
            args.ground_truth_root / "normals_gt" / f"val_normalCam_{frame:04d}.npy"
        )
        required = (normal_path, depth_target_path, normal_target_path)
        missing = [path for path in required if not path.exists()]
        if missing and args.skip_missing:
            print(
                f"Skipping frame {frame:04d}; missing "
                + ", ".join(str(path) for path in missing)
            )
            continue
        if missing:
            raise FileNotFoundError(
                "missing evaluation input(s): "
                + ", ".join(str(path) for path in missing)
            )

        depth_prediction_raw = np.load(depth_path)
        normal_prediction_raw = np.load(normal_path)
        depth_target_raw = np.load(depth_target_path)
        normal_target_raw = np.load(normal_target_path)
        target_depth_shape = np.asarray(depth_target_raw).squeeze().shape
        if len(target_depth_shape) != 2:
            raise ValueError(
                f"ground-truth depth must be [H, W], got {target_depth_shape}"
            )
        shape = infer_map_shape(depth_prediction_raw, target_depth_shape, channels=1)
        normal_shape = infer_map_shape(
            normal_prediction_raw, target_depth_shape, channels=3
        )
        if normal_shape != shape:
            raise ValueError(
                f"depth and normal resolutions differ: {shape} and {normal_shape}"
            )

        depth_prediction = _prediction_map(depth_prediction_raw, shape, 1)
        normal_prediction = _prediction_map(normal_prediction_raw, shape, 3)
        depth_target = resize_depth_target(depth_target_raw, shape)
        normal_target = resize_normal_target(normal_target_raw, shape)
        depth_result = depth_metrics(
            depth_prediction,
            depth_target,
            alignment=args.depth_alignment,
            max_depth=args.max_depth,
        )
        normal_result = normal_metrics(
            normal_prediction,
            normal_target,
            valid_depth=depth_target > 1e-6,
        )
        row: dict[str, object] = {
            "frame": frame,
            "prediction": depth_path.name,
            **{f"depth_{key}": value for key, value in depth_result.items()},
            **{f"normal_{key}": value for key, value in normal_result.items()},
        }
        rows.append(row)
        print(
            f"[{len(rows):04d}] frame={frame:04d} "
            f"AbsRel={depth_result['abs_rel']:.4f} "
            f"RMSE={depth_result['rmse']:.4f} "
            f"normal_MAE={normal_result['mae_degrees']:.2f} deg"
        )

    if not rows:
        raise RuntimeError("no complete depth/normal prediction pairs were evaluated")
    aggregate = {
        "normal_mae_degrees": _mean_metrics(rows, "normal_mae_degrees"),
        "normal_accuracy_11_25_percent": _mean_metrics(
            rows, "normal_accuracy_11_25_percent"
        ),
        "normal_accuracy_22_5_percent": _mean_metrics(
            rows, "normal_accuracy_22_5_percent"
        ),
        "depth_abs_rel": _mean_metrics(rows, "depth_abs_rel"),
        "depth_rmse": _mean_metrics(rows, "depth_rmse"),
        "depth_rmse_depth_units": _mean_metrics(rows, "depth_rmse_depth_units"),
        "depth_delta_1_25_percent": _mean_metrics(rows, "depth_delta_1_25_percent"),
    }
    report: dict[str, object] = {
        "protocol": {
            "aggregation": "macro-average over views",
            "scene": scene,
            "view_split": (
                "all prediction files"
                if selected_frames is None
                else str(args.view_split.resolve())
            ),
            "view_indices": selected_frames,
            "depth_alignment": args.depth_alignment,
            "depth_range": [1e-6, args.max_depth],
            "rmse": "per-view RMSE divided by mean valid ground-truth depth",
            "normal_coordinates": "camera",
        },
        "prediction_root": str(args.prediction_root.resolve()),
        "ground_truth_root": str(args.ground_truth_root.resolve()),
        "evaluated_views": len(rows),
        "aggregate": aggregate,
        "views": rows,
    }
    print(
        "\n"
        f"Normal: MAE={aggregate['normal_mae_degrees']:.2f} deg, "
        f"<11.25={aggregate['normal_accuracy_11_25_percent']:.2f}%, "
        f"<22.5={aggregate['normal_accuracy_22_5_percent']:.2f}%\n"
        f"Depth: AbsRel={aggregate['depth_abs_rel']:.4f}, "
        f"RMSE={aggregate['depth_rmse']:.4f}, "
        f"delta<1.25={aggregate['depth_delta_1_25_percent']:.2f}%"
    )
    return report


def evaluate_mesh(args: argparse.Namespace) -> dict[str, object]:
    print(f"Sampling reconstruction: {args.prediction}")
    reconstruction = sample_mesh(
        args.prediction,
        spacing=args.sample_spacing,
        seed=args.seed,
        workers=args.workers,
    )
    print(f"Sampling ground truth: {args.ground_truth}")
    target = sample_mesh(
        args.ground_truth,
        spacing=args.sample_spacing,
        seed=args.seed,
        workers=args.workers,
    )
    metrics = point_cloud_metrics(
        reconstruction,
        target,
        max_distance=args.max_distance,
        fscore_thresholds=args.fscore_thresholds,
    )
    report: dict[str, object] = {
        "protocol": {
            "sample_spacing_meters": args.sample_spacing,
            "maximum_chamfer_distance_meters": args.max_distance,
            "paper_chamfer_unit": "decimeters",
        },
        "prediction": str(args.prediction.resolve()),
        "ground_truth": str(args.ground_truth.resolve()),
        "metrics": metrics,
    }
    one_centimeter = metrics["fscores"].get("10mm")  # type: ignore[union-attr]
    f1_text = ""
    if one_centimeter is not None:
        f1_text = f", F1@1cm={one_centimeter['f1']:.4f}"
    print(
        f"Chamfer={metrics['chamfer_meters']:.6f} m "
        f"({metrics['chamfer_decimeters']:.6f} dm){f1_text}"
    )
    return report


def summarize_reports(args: argparse.Namespace) -> dict[str, object]:
    reports = []
    for path in args.reports:
        report = json.loads(path.read_text())
        if "aggregate" not in report:
            raise ValueError(f"map evaluation report has no aggregate section: {path}")
        reports.append((path, report))
    keys = (
        "normal_mae_degrees",
        "normal_accuracy_11_25_percent",
        "normal_accuracy_22_5_percent",
        "depth_abs_rel",
        "depth_rmse",
        "depth_rmse_depth_units",
        "depth_delta_1_25_percent",
    )
    aggregate = {
        key: float(np.mean([float(report["aggregate"][key]) for _, report in reports]))
        for key in keys
    }
    result: dict[str, object] = {
        "protocol": "macro-average over scene-level map reports",
        "evaluated_scenes": len(reports),
        "aggregate": aggregate,
        "scenes": [
            {
                "report": str(path.resolve()),
                "evaluated_views": report.get("evaluated_views"),
                "aggregate": report["aggregate"],
            }
            for path, report in reports
        ],
    }
    print(
        f"Normal: MAE={aggregate['normal_mae_degrees']:.2f} deg, "
        f"<11.25={aggregate['normal_accuracy_11_25_percent']:.2f}%, "
        f"<22.5={aggregate['normal_accuracy_22_5_percent']:.2f}%\n"
        f"Depth: AbsRel={aggregate['depth_abs_rel']:.4f}, "
        f"RMSE={aggregate['depth_rmse']:.4f}, "
        f"delta<1.25={aggregate['depth_delta_1_25_percent']:.2f}%"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate GLINT geometry on the 3D-FRONT-T benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    maps = subparsers.add_parser("maps", help="Evaluate depth and normal maps.")
    maps.add_argument(
        "--prediction-root",
        type=Path,
        required=True,
        help="Directory containing depths_pred/ and normals_pred/.",
    )
    maps.add_argument("--ground-truth-root", type=Path, required=True)
    maps.add_argument("--output", type=Path, required=True)
    maps.add_argument(
        "--scene",
        help="Scene key in the view split; defaults to the GT directory name.",
    )
    maps.add_argument("--view-split", type=Path, default=DEFAULT_VIEW_SPLIT)
    maps.add_argument(
        "--all-views",
        action="store_true",
        help="Ignore the official 3D-FRONT-T view split and use every prediction.",
    )
    maps.add_argument("--depth-alignment", choices=("median", "none"), default="median")
    maps.add_argument("--max-depth", type=float, default=20.0)
    maps.add_argument(
        "--skip-missing",
        action="store_true",
        help="Evaluate only complete prediction/ground-truth pairs.",
    )
    maps.set_defaults(evaluate=evaluate_maps)

    mesh = subparsers.add_parser("mesh", help="Evaluate a reconstructed mesh.")
    mesh.add_argument("--prediction", type=Path, required=True)
    mesh.add_argument("--ground-truth", type=Path, required=True)
    mesh.add_argument("--output", type=Path, required=True)
    mesh.add_argument("--sample-spacing", type=float, default=0.015)
    mesh.add_argument("--max-distance", type=float, default=10.0)
    mesh.add_argument("--fscore-thresholds", type=float, nargs="+", default=(0.01,))
    mesh.add_argument("--seed", type=int, default=0)
    mesh.add_argument(
        "--workers",
        type=int,
        help="Mesh-sampling processes; defaults to the available CPU count.",
    )
    mesh.set_defaults(evaluate=evaluate_mesh)

    summary = subparsers.add_parser(
        "summary", help="Average map evaluation reports over scenes."
    )
    summary.add_argument("--reports", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.set_defaults(evaluate=summarize_reports)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "maps" and args.max_depth <= 0.0:
        parser.error("--max-depth must be positive")
    if args.command == "mesh" and args.workers is not None and args.workers <= 0:
        parser.error("--workers must be positive")
    report = args.evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
