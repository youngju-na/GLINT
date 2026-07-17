# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from glint.geometry_metrics import (
    depth_metrics,
    infer_map_shape,
    normal_metrics,
    point_cloud_metrics,
    resize_depth_target,
    resize_normal_target,
)


def test_normal_metrics_use_angular_thresholds():
    target = np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    prediction = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

    metrics = normal_metrics(prediction, target)

    assert metrics["mae_degrees"] == pytest.approx(45.0)
    assert metrics["accuracy_11_25_percent"] == pytest.approx(50.0)
    assert metrics["accuracy_22_5_percent"] == pytest.approx(50.0)
    assert metrics["valid_pixels"] == 2


def test_depth_metrics_median_alignment_and_normalized_rmse():
    target = np.array([[1.0, 2.0], [3.0, 4.0]])
    prediction = 2.0 * target

    aligned = depth_metrics(prediction, target, alignment="median")
    unaligned = depth_metrics(prediction, target, alignment="none")

    assert aligned["alignment_scale"] == pytest.approx(0.5)
    assert aligned["abs_rel"] == pytest.approx(0.0)
    assert aligned["rmse"] == pytest.approx(0.0)
    assert aligned["delta_1_25_percent"] == pytest.approx(100.0)
    expected_raw_rmse = np.sqrt(np.mean(target**2))
    assert unaligned["rmse_depth_units"] == pytest.approx(expected_raw_rmse)
    assert unaligned["rmse"] == pytest.approx(expected_raw_rmse / target.mean())


def test_resize_targets_match_released_half_resolution_protocol():
    depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    normals = np.arange(48, dtype=np.float32).reshape(4, 4, 3)

    resized_depth = resize_depth_target(depth, (2, 2))
    resized_normals = resize_normal_target(normals, (2, 2))

    assert np.allclose(resized_depth, [[2.5, 4.5], [10.5, 12.5]])
    assert np.array_equal(resized_normals, normals[::2, ::2])
    assert infer_map_shape(np.zeros((1, 4, 1)), (4, 4), channels=1) == (2, 2)
    assert infer_map_shape(np.zeros((1, 4, 3)), (4, 4), channels=3) == (2, 2)


def test_point_cloud_metrics_report_paper_units_and_fscore():
    target = np.array([[0.0, 0.0, 0.0]])
    reconstruction = np.array([[0.1, 0.0, 0.0]])

    metrics = point_cloud_metrics(
        reconstruction,
        target,
        max_distance=1.0,
        fscore_thresholds=(0.05, 0.2),
    )

    assert metrics["chamfer_meters"] == pytest.approx(0.1)
    assert metrics["chamfer_decimeters"] == pytest.approx(1.0)
    assert metrics["fscores"]["50mm"]["f1"] == pytest.approx(0.0)
    assert metrics["fscores"]["200mm"]["f1"] == pytest.approx(1.0)
