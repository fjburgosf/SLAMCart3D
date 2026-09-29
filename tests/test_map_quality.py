import numpy as np
import pytest

from slamcart3d.metrics.direction import failure_margin, success_relation
from slamcart3d.metrics.map_quality import point_map_metrics, sample_wall_points
from slamcart3d.worlds.geometry import WallSegment, World3D


def test_perfect_sampled_map_has_zero_distance_and_full_coverage() -> None:
    world = World3D("wall", (WallSegment(0.0, 0.0, 1.0, 0.0),))
    points = sample_wall_points(world, 0.1)
    metrics = point_map_metrics(points, world, spacing_m=0.1)
    assert metrics["map_accuracy_mean_m"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["map_completeness_mean_m"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["map_chamfer_mean_m"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["map_precision_0p25"] == pytest.approx(1.0)
    assert metrics["map_recall_0p25"] == pytest.approx(1.0)
    assert metrics["map_f1_0p25"] == pytest.approx(1.0)


def test_outlier_reduces_precision_and_empty_map_has_zero_coverage() -> None:
    world = World3D("wall", (WallSegment(0.0, 0.0, 1.0, 0.0),))
    points = np.vstack((sample_wall_points(world, 0.1), np.array([[10.0, 10.0]])))
    metrics = point_map_metrics(points, world)
    assert metrics["map_precision_0p25"] < 1.0
    assert metrics["map_recall_0p25"] == pytest.approx(1.0)
    assert metrics["map_accuracy_mean_m"] > 0.0
    empty = point_map_metrics(np.empty((0, 2)), world)
    assert empty["map_recall_0p25"] == 0.0
    assert "map_chamfer_mean_m" not in empty


def test_frontier_margin_has_correct_direction_for_map_quality() -> None:
    assert failure_margin(0.8, "map_f1_0p25", 0.9) == pytest.approx(0.1)
    assert failure_margin(0.95, "map_f1_0p25", 0.9) == pytest.approx(-0.05)
    assert failure_margin(0.3, "ate_rmse_m", 0.2) == pytest.approx(0.1)
    assert success_relation("map_f1_0p25") == ">"
    with pytest.raises(ValueError, match="unsupported frontier metric"):
        failure_margin(0.2, "unknown", 0.1)
