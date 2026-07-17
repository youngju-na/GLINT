# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Geometry metrics used for the 3D-FRONT-T evaluation."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


def normal_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_depth: np.ndarray | None = None,
    *,
    eps: float = 1e-6,
) -> dict[str, float | int]:
    """Compute angular normal errors in degrees and threshold accuracies."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "normal maps must have matching [H, W, 3] shapes, got "
            f"{prediction.shape} and {target.shape}"
        )
    if prediction.shape[-1] != 3:
        raise ValueError(
            f"normal maps must have three channels, got {prediction.shape}"
        )

    target_norm = np.linalg.norm(target, axis=-1)
    prediction_norm = np.linalg.norm(prediction, axis=-1)
    valid = np.isfinite(target).all(axis=-1)
    valid &= np.isfinite(prediction).all(axis=-1)
    valid &= target_norm > eps
    if valid_depth is not None:
        valid_depth = np.asarray(valid_depth)
        if valid_depth.shape != target.shape[:2]:
            raise ValueError(
                f"valid_depth must be {target.shape[:2]}, got {valid_depth.shape}"
            )
        valid &= valid_depth.astype(bool)
    if not valid.any():
        raise ValueError("normal maps contain no valid ground-truth pixels")

    prediction_unit = prediction[valid] / np.maximum(
        prediction_norm[valid][:, None], eps
    )
    target_unit = target[valid] / np.maximum(target_norm[valid][:, None], eps)
    cosine = np.sum(prediction_unit * target_unit, axis=-1).clip(-1.0, 1.0)
    error = np.rad2deg(np.arccos(cosine))
    return {
        "mae_degrees": float(error.mean()),
        "accuracy_11_25_percent": float((error < 11.25).mean() * 100.0),
        "accuracy_22_5_percent": float((error < 22.5).mean() * 100.0),
        "valid_pixels": int(error.size),
    }


def depth_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    alignment: str = "median",
    min_depth: float = 1e-6,
    max_depth: float = 20.0,
) -> dict[str, float | int]:
    """Compute the depth metrics reported for GLINT.

    The paper protocol aligns each predicted view to the ground truth by a
    single median scale. RMSE is divided by the mean valid ground-truth depth,
    making it comparable across scenes with different metric scales. The raw
    RMSE in input depth units is returned as ``rmse_depth_units`` as well.
    """

    prediction = np.asarray(prediction, dtype=np.float64).squeeze()
    target = np.asarray(target, dtype=np.float64).squeeze()
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            "depth maps must have matching [H, W] shapes, got "
            f"{prediction.shape} and {target.shape}"
        )
    if alignment not in {"none", "median"}:
        raise ValueError(f"unsupported depth alignment: {alignment}")

    valid = np.isfinite(target) & np.isfinite(prediction)
    valid &= (target > min_depth) & (prediction > 0.0)
    valid &= (target < max_depth) & (prediction < max_depth)
    if not valid.any():
        raise ValueError("depth maps contain no jointly valid pixels")

    scale = 1.0
    if alignment == "median":
        prediction_median = float(np.median(prediction[valid]))
        if prediction_median <= 0.0:
            raise ValueError("the median valid predicted depth is not positive")
        scale = float(np.median(target[valid]) / prediction_median)

    predicted = prediction[valid] * scale
    expected = target[valid]
    difference = predicted - expected
    ratio = np.maximum(predicted / expected, expected / predicted)
    rmse_depth_units = float(np.sqrt(np.mean(difference**2)))
    mean_target_depth = float(expected.mean())
    return {
        "abs_rel": float(np.mean(np.abs(difference) / expected)),
        "rmse": rmse_depth_units / mean_target_depth,
        "rmse_depth_units": rmse_depth_units,
        "delta_1_25_percent": float((ratio < 1.25).mean() * 100.0),
        "alignment_scale": scale,
        "valid_pixels": int(expected.size),
    }


def resize_depth_target(target: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Area-average a ground-truth depth map to the prediction resolution."""

    target = np.asarray(target).squeeze()
    if target.ndim != 2:
        raise ValueError(f"target depth must be [H, W], got {target.shape}")
    if target.shape == shape:
        return target.astype(np.float32, copy=False)
    source_height, source_width = target.shape
    height, width = shape
    if source_height % height or source_width % width:
        raise ValueError(
            f"cannot area-average depth from {target.shape} to {shape}; "
            "both scale factors must be integers"
        )
    scale_y, scale_x = source_height // height, source_width // width
    return (
        target.astype(np.float32)
        .reshape(height, scale_y, width, scale_x)
        .mean(axis=(1, 3))
    )


def resize_normal_target(target: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Subsample camera-space ground-truth normals to the prediction resolution."""

    target = np.asarray(target).squeeze()
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError(f"target normals must be [H, W, 3], got {target.shape}")
    if target.shape[:2] == shape:
        return target.astype(np.float32, copy=False)
    source_height, source_width = target.shape[:2]
    height, width = shape
    if source_height % height or source_width % width:
        raise ValueError(
            f"cannot subsample normals from {target.shape[:2]} to {shape}; "
            "both scale factors must be integers"
        )
    scale_y, scale_x = source_height // height, source_width // width
    return target[::scale_y, ::scale_x].astype(np.float32, copy=False)


def infer_map_shape(
    value: np.ndarray,
    target_shape: tuple[int, int],
    *,
    channels: int,
) -> tuple[int, int]:
    """Infer an image shape from a saved dense or flattened prediction."""

    value = np.asarray(value).squeeze()
    if channels == 1 and value.ndim == 2 and value.shape[-1] != 1:
        return int(value.shape[0]), int(value.shape[1])
    if channels == 3 and value.ndim == 3 and value.shape[-1] == 3:
        return int(value.shape[0]), int(value.shape[1])
    elements = value.size // channels
    if elements * channels != value.size:
        raise ValueError(
            f"prediction shape {value.shape} is not divisible by {channels}"
        )
    target_height, target_width = target_shape
    for scale in range(1, 33):
        if target_height % scale or target_width % scale:
            continue
        height, width = target_height // scale, target_width // scale
        if height * width == elements:
            return height, width
    raise ValueError(
        f"cannot infer a prediction resolution with {elements} pixels from "
        f"ground-truth shape {target_shape}"
    )


def point_cloud_metrics(
    reconstruction: np.ndarray,
    target: np.ndarray,
    *,
    max_distance: float = 10.0,
    fscore_thresholds: Iterable[float] = (0.01,),
) -> dict[str, object]:
    """Compute symmetric Chamfer distance and point-cloud F-scores."""

    reconstruction = _validate_points(reconstruction, "reconstruction")
    target = _validate_points(target, "target")
    if max_distance <= 0.0:
        raise ValueError("max_distance must be positive")

    reconstruction_to_target = cKDTree(target).query(reconstruction, k=1, workers=-1)[0]
    target_to_reconstruction = cKDTree(reconstruction).query(target, k=1, workers=-1)[0]
    reconstruction_inlier = reconstruction_to_target < max_distance
    target_inlier = target_to_reconstruction < max_distance
    if not reconstruction_inlier.any() or not target_inlier.any():
        raise ValueError("max_distance rejected every point in one direction")

    accuracy = float(reconstruction_to_target[reconstruction_inlier].mean())
    completeness = float(target_to_reconstruction[target_inlier].mean())
    chamfer = 0.5 * (accuracy + completeness)
    fscores: dict[str, dict[str, float]] = {}
    for threshold in fscore_thresholds:
        threshold = float(threshold)
        if threshold <= 0.0:
            raise ValueError("F-score thresholds must be positive")
        precision = float((reconstruction_to_target < threshold).mean())
        recall = float((target_to_reconstruction < threshold).mean())
        fscore = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        fscores[f"{threshold * 1000:g}mm"] = {
            "threshold_meters": threshold,
            "precision": precision,
            "recall": recall,
            "f1": fscore,
        }
    return {
        "accuracy_meters": accuracy,
        "completeness_meters": completeness,
        "chamfer_meters": chamfer,
        "chamfer_decimeters": chamfer * 10.0,
        "fscores": fscores,
        "reconstruction_points": int(len(reconstruction)),
        "target_points": int(len(target)),
    }


def sample_mesh(
    path: str | Path,
    *,
    spacing: float = 0.015,
    seed: int = 0,
    workers: int | None = None,
) -> np.ndarray:
    """Sample and radius-downsample a triangle mesh as in the GLINT protocol."""

    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    try:
        import open3d as o3d
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError(
            "mesh evaluation requires requirements-geometry.txt"
        ) from exc

    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.size == 0 or triangles.size == 0:
        raise ValueError(f"{path} is not a non-empty triangle mesh")

    triangle_vertices = vertices[triangles]
    edge_1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge_2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    length_1 = np.linalg.norm(edge_1, axis=-1)
    length_2 = np.linalg.norm(edge_2, axis=-1)
    twice_area = np.linalg.norm(np.cross(edge_1, edge_2), axis=-1)
    valid = twice_area > 0.0
    edge_1, edge_2, triangle_vertices = (
        edge_1[valid],
        edge_2[valid],
        triangle_vertices[valid],
    )
    threshold = spacing * np.sqrt(length_1[valid] * length_2[valid] / twice_area[valid])
    count_1 = np.floor(length_1[valid] / threshold).astype(np.int64)
    count_2 = np.floor(length_2[valid] / threshold).astype(np.int64)
    tasks = (
        (
            count_1[index],
            count_2[index],
            edge_1[index],
            edge_2[index],
            triangle_vertices[index, 0],
        )
        for index in range(len(count_1))
    )
    if workers == 1:
        sampled = list(map(_sample_triangle, tasks))
    else:
        with mp.Pool(processes=workers) as pool:
            sampled = pool.map(_sample_triangle, tasks, chunksize=1024)
    points = np.concatenate((vertices, *sampled), axis=0)

    rng = np.random.default_rng(seed)
    rng.shuffle(points, axis=0)
    neighbors = NearestNeighbors(radius=spacing, algorithm="kd_tree", n_jobs=-1)
    neighbors.fit(points)
    neighborhoods = neighbors.radius_neighbors(
        points, radius=spacing, return_distance=False
    )
    keep = np.ones(len(points), dtype=bool)
    for index, indices in enumerate(neighborhoods):
        if keep[index]:
            keep[indices] = False
            keep[index] = True
    return points[keep]


def _sample_triangle(
    task: tuple[int, int, np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    count_1, count_2, edge_1, edge_2, origin = task
    coordinates = np.mgrid[: count_1 + 1, : count_2 + 1].astype(np.float64)
    coordinates += 0.5
    coordinates[0] /= max(count_1, 1e-7)
    coordinates[1] /= max(count_2, 1e-7)
    coordinates = np.transpose(coordinates, (1, 2, 0))
    coordinates = coordinates[coordinates.sum(axis=-1) < 1.0]
    return (
        origin[None]
        + edge_1[None] * coordinates[:, :1]
        + edge_2[None] * coordinates[:, 1:]
    )


def _validate_points(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[-1] != 3 or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty [N, 3] array, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite points")
    return value
