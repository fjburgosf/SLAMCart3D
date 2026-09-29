"""Objective static-wall 3D map metrics and their limitations."""

import numpy as np
import pytest

from slamcart3d.metrics.map_quality_3d import sample_wall_surface_points, wall_point_map_metrics_3d
from slamcart3d.worlds.geometry import WallSegment, World3D


def test_exact_wall_surface_cloud_scores_perfectly() -> None:
    world = World3D("single wall", (WallSegment(5.0, 0.0, 5.0, 2.0, height_m=2.0),))
    points = sample_wall_surface_points(world)
    metrics = wall_point_map_metrics_3d(points, world)
    assert metrics["map3d_wall_accuracy_mean_m"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["map3d_wall_completeness_mean_m"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["map3d_wall_precision_0p25"] == 1.0
    assert metrics["map3d_wall_recall_0p25"] == 1.0
    assert metrics["map3d_wall_f1_0p25"] == 1.0


def test_ground_like_points_are_excluded_but_wall_outlier_is_penalized() -> None:
    world = World3D("single wall", (WallSegment(5.0, 0.0, 5.0, 2.0, height_m=2.0),))
    reference = sample_wall_surface_points(world)
    baseline = wall_point_map_metrics_3d(reference, world)
    ground_augmented = wall_point_map_metrics_3d(
        np.vstack((reference, np.array([[100.0, 100.0, 0.0]]))), world
    )
    assert ground_augmented == baseline
    outlier = wall_point_map_metrics_3d(np.vstack((reference, np.array([[100.0, 100.0, 1.0]]))), world)
    assert outlier["map3d_wall_precision_0p25"] < 1.0
    assert outlier["map3d_wall_accuracy_mean_m"] > 0.0


def test_empty_cloud_has_zero_finite_3d_wall_scores() -> None:
    world = World3D("single wall", (WallSegment(5.0, 0.0, 5.0, 2.0),))
    metrics = wall_point_map_metrics_3d(np.empty((0, 3)), world)
    assert metrics == {
        "map3d_wall_precision_0p25": 0.0,
        "map3d_wall_recall_0p25": 0.0,
        "map3d_wall_f1_0p25": 0.0,
    }


def test_3d_wall_metric_rejects_invalid_geometry_and_points() -> None:
    world = World3D("single wall", (WallSegment(5.0, 0.0, 5.0, 2.0),))
    with pytest.raises(ValueError, match="N x 3"):
        wall_point_map_metrics_3d(np.empty((0, 2)), world)
    with pytest.raises(ValueError, match="spacing"):
        sample_wall_surface_points(world, spacing_m=0.0)
    with pytest.raises(ValueError, match="tolerance"):
        wall_point_map_metrics_3d(np.empty((0, 3)), world, tolerance_m=0.0)
