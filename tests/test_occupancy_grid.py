"""Observed occupancy is distinct from unknown space and static reference walls."""

import csv
import json

import numpy as np
import pytest

from slamcart3d.core.config import ExperimentConfig, LidarConfig, OccupancyGridConfig, SimulationConfig
from slamcart3d.core.models import LidarScan, Pose2D
from slamcart3d.experiments.experiment import Experiment
from slamcart3d.metrics.occupancy_grid import OccupancyGrid
from slamcart3d.worlds.geometry import WallSegment, World3D


def test_hit_marks_endpoint_occupied_and_only_ray_interior_free() -> None:
    grid = OccupancyGrid(resolution_m=0.2, beam_stride=1)
    world = World3D("one_wall", (WallSegment(2.0, -1.0, 2.0, 1.0),))
    grid.observe(LidarScan(0.0, np.array([0.0]), np.array([2.0])), Pose2D(0.0, 0.0, 0.0), 10.0)
    assert (9, 0) in grid.reference_occupied(world)
    assert (10, 0) in grid.reference_occupied(world)
    assert (10, 0) in grid.occupied
    assert (10, 0) not in grid.free
    assert (0, 0) in grid.free
    metrics = grid.metrics(world)
    assert metrics["occupancy_occupied_precision"] == pytest.approx(1.0)
    assert 0.0 < metrics["occupancy_occupied_recall"] < 1.0
    assert metrics["occupancy_observed_free_accuracy"] == pytest.approx(1.0)
    assert any(row[4] == "unknown" and row[5] == "occupied" for row in grid.rows(world))


def test_missing_beam_does_not_invent_free_space_and_false_hit_lowers_precision() -> None:
    grid = OccupancyGrid(resolution_m=0.2, beam_stride=1)
    world = World3D("one_wall", (WallSegment(2.0, -1.0, 2.0, 1.0),))
    grid.observe(LidarScan(0.0, np.array([0.0]), np.array([np.nan])), Pose2D(0.0, 0.0, 0.0), 10.0)
    assert not grid.free and not grid.occupied
    assert grid.metrics(world)["occupancy_occupied_recall"] == 0.0
    grid.observe(LidarScan(0.1, np.array([0.0]), np.array([1.0])), Pose2D(0.0, 0.0, 0.0), 10.0)
    assert grid.metrics(world)["occupancy_occupied_precision"] == 0.0


def test_occupancy_export_contains_unknown_reference_cells(tmp_path) -> None:
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=12),
        simulation=SimulationConfig(duration_s=0.3, timestep_s=0.1),
        occupancy_grid=OccupancyGridConfig(enabled=True, resolution_m=0.2, beam_stride=2),
    )
    first = Experiment(config).run()
    second = Experiment(config).run()
    assert first.metrics["occupancy_occupied_iou"] == second.metrics["occupancy_occupied_iou"]
    assert 0.0 <= first.metrics["occupancy_occupied_iou"] <= 1.0
    target = Experiment(config).run_and_export(tmp_path, plot=True)
    with (target / "map" / "occupancy_grid.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert {row["estimated"] for row in rows} >= {"occupied", "free", "unknown"}
    assert (target / "figures" / "occupancy_grid.png").is_file()
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["occupancy_evaluation"]["resolution_m"] == 0.2


def test_invalid_occupancy_resolution_and_stride_are_rejected() -> None:
    with pytest.raises(ValueError, match="resolution"):
        ExperimentConfig(occupancy_grid=OccupancyGridConfig(enabled=True, resolution_m=0.0))
    with pytest.raises(ValueError, match="beam stride"):
        ExperimentConfig(occupancy_grid=OccupancyGridConfig(enabled=True, beam_stride=0))
