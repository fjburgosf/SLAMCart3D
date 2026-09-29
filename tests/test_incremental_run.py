"""Incremental persistence and editable scene regression checks."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from slamcart3d.core.config import ExperimentConfig, LidarConfig, SimulationConfig, WallConfig
from slamcart3d.gui.viewer3d import cart_vertices
from slamcart3d.io.export import IncrementalRunWriter, export_result
from slamcart3d.simulation.runner import run_simulation


def test_incremental_files_are_flushed_at_each_step(tmp_path) -> None:
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.3, timestep_s=0.1),
    )
    writer = None

    def on_start(started):
        nonlocal writer
        writer = IncrementalRunWriter(started, tmp_path)
        assert json.loads((writer.target / "status.json").read_text())["completed_steps"] == 0

    def on_step(step):
        assert writer is not None
        writer.append(step)
        status = json.loads((writer.target / "status.json").read_text())
        assert status["completed_steps"] == step.index
        with (writer.target / "trajectory_live.csv").open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == step.index

    result = run_simulation(config, on_start=on_start, on_step=on_step)
    assert writer is not None
    writer.close()
    assert export_result(result, tmp_path, reuse_existing=True) == writer.target
    writer.complete()
    assert json.loads((writer.target / "status.json").read_text())["state"] == "complete"
    with (writer.target / "point_cloud_live.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == len(result.map_points)


def test_custom_3d_world_round_trips_and_is_used_by_simulation() -> None:
    walls = (
        WallConfig(0, 0, 8, 0, 2.0),
        WallConfig(8, 0, 8, 8, 3.0),
        WallConfig(8, 8, 0, 8, 2.5),
        WallConfig(0, 8, 0, 0, 2.5),
    )
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.1, timestep_s=0.1, world="custom", world_segments=walls),
    )
    restored = ExperimentConfig.from_dict(config.to_dict())
    assert restored == config
    result = run_simulation(restored)
    assert result.world.name == "custom"
    assert [wall.height_m for wall in result.world.segments] == [2.0, 3.0, 2.5, 2.5]


def test_invalid_custom_walls_are_rejected() -> None:
    with pytest.raises(ValueError, match="requires at least one"):
        ExperimentConfig(simulation=SimulationConfig(world="custom"))
    with pytest.raises(ValueError, match="positive height"):
        ExperimentConfig(
            simulation=SimulationConfig(world="custom", world_segments=(WallConfig(0, 0, 1, 0, 0),))
        )


def test_cart_vertices_follow_heading() -> None:
    east = cart_vertices(np.array([5.0, 4.0, 0.0]))
    north = cart_vertices(np.array([5.0, 4.0, np.pi / 2]))
    assert east.shape == (8, 3)
    assert east[:, 2].min() == pytest.approx(0.08)
    assert east[:, 2].max() == pytest.approx(0.43)
    assert east[1, 0] > east[0, 0]
    assert north[1, 1] > north[0, 1]


def test_cart_stops_before_outer_wall_and_can_disable_collision_model() -> None:
    simulation = SimulationConfig(
        duration_s=20.0,
        timestep_s=0.1,
        linear_velocity_m_s=0.4,
        yaw_rate_rad_s=0.0,
    )
    config = ExperimentConfig(lidar=LidarConfig(beam_count=8), simulation=simulation)
    blocked = run_simulation(config)
    assert blocked.ground_truth[-1, 0] < 10.0
    assert blocked.metrics["wall_blocked_steps"] > 0
    assert blocked.metrics["ground_truth_static_wall_contact_steps"] == 0
    unblocked = run_simulation(
        ExperimentConfig(lidar=config.lidar, simulation=replace(simulation, stop_at_walls=False))
    )
    assert unblocked.ground_truth[-1, 0] > 10.0
    assert unblocked.metrics["wall_blocked_steps"] == 0
