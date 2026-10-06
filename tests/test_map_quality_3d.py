"""Objective static-wall 3D map metrics and their limitations."""

import json
from dataclasses import replace

import numpy as np
import pytest

from slamcart3d.core.config import ExperimentConfig, WallConfig
from slamcart3d.io.export import export_result
from slamcart3d.metrics.map_quality_3d import sample_wall_surface_points, wall_point_map_metrics_3d
from slamcart3d.simulation.runner import run_simulation
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


@pytest.mark.parametrize("height", [0.05, 0.1, 0.1000000005])
def test_low_custom_walls_export_without_inventing_3d_quality_scores(tmp_path, height) -> None:
    base = ExperimentConfig()
    config = replace(
        base,
        lidar=replace(base.lidar, beam_count=8),
        lidar3d=replace(base.lidar3d, enabled=True, azimuth_beams=3, elevation_beams=3),
        simulation=replace(
            base.simulation,
            duration_s=0.2,
            world="custom",
            world_segments=(WallConfig(9.0, 0.0, 9.0, 8.0, height),),
        ),
    )
    result = run_simulation(config)
    assert all(np.isfinite(value) for value in result.metrics.values())
    assert not any(name.startswith("map3d_wall_") for name in result.metrics)
    assert result.map_points_3d.shape[1] == 3
    path = export_result(result, tmp_path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["map3d_evaluation"]["available"] is False
    assert "ground exclusion height" in metadata["map3d_evaluation"]["unavailable_reason"]
