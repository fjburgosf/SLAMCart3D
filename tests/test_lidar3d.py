"""Finite-height 3D ray casts and optional multilayer dataset export."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from slamcart3d.core.config import ExperimentConfig, Lidar3DConfig, LidarConfig, SimulationConfig
from slamcart3d.core.models import Pose2D
from slamcart3d.datasets.generator import generate_dataset
from slamcart3d.io.export import export_result
from slamcart3d.sensors.lidar3d import MultiLayerLidar
from slamcart3d.simulation.observations import ObservationStream
from slamcart3d.simulation.runner import run_simulation
from slamcart3d.worlds.geometry import WallSegment, World3D


def test_3d_ray_respects_wall_height_and_ground() -> None:
    world = World3D("test", (WallSegment(5.0, -2.0, 5.0, 2.0, height_m=2.0),))
    assert world.ray_distance_3d(0.0, 0.0, 1.0, 0.0, 0.0, 10.0) == pytest.approx(5.0)
    assert world.ray_distance_3d(0.0, 0.0, 1.0, 0.0, np.arctan(2.0 / 5.0), 10.0) == 10.0
    assert world.ray_distance_3d(0.0, 0.0, 1.0, 0.0, -np.pi / 4.0, 10.0) == pytest.approx(np.sqrt(2.0))


def test_multilayer_lidar_creates_nonplanar_point_cloud() -> None:
    world = World3D("test", (WallSegment(5.0, -2.0, 5.0, 2.0, height_m=2.0),))
    lidar = MultiLayerLidar(
        Lidar3DConfig(enabled=True, azimuth_beams=3, elevation_beams=3, horizontal_fov_deg=90.0)
    )
    scan = lidar.scan(0.0, Pose2D(0.0, 0.0, 0.0), world)
    assert len(scan.ranges) == 9
    assert scan.ranges[4] == pytest.approx(5.0)
    cloud = lidar.point_cloud(scan, Pose2D(0.0, 0.0, 0.0))
    assert cloud.shape[1] == 3
    assert np.ptp(cloud[:, 2]) > 0.1
    assert np.min(cloud[:, 2]) >= -1e-12


def test_optional_3d_sensor_leaves_planar_observations_unchanged_and_exports(tmp_path) -> None:
    base = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )
    enabled = replace(base, lidar3d=Lidar3DConfig(enabled=True, azimuth_beams=3, elevation_beams=3))
    plain_frames = list(ObservationStream(base).frames())
    enhanced_frames = list(ObservationStream(enabled).frames())
    for plain, enhanced in zip(plain_frames, enhanced_frames, strict=True):
        np.testing.assert_allclose(
            plain.measured_lidar.ranges, enhanced.measured_lidar.ranges, equal_nan=True
        )
        assert plain.measured_left_rate_rad_s == enhanced.measured_left_rate_rad_s
        assert plain.measured_right_rate_rad_s == enhanced.measured_right_rate_rad_s
        assert plain.ideal_lidar3d is None
        assert enhanced.ideal_lidar3d is not None
    target = generate_dataset(enabled, tmp_path)
    with (target / "lidar3d.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18
    assert any(row["z_world_m"] for row in rows)
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["lidar3d_rays_per_frame"] == 9


def test_run_accumulates_measured_3d_points_at_estimated_poses(tmp_path) -> None:
    base = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )
    enabled = replace(base, lidar3d=Lidar3DConfig(enabled=True, azimuth_beams=3, elevation_beams=3))
    result = run_simulation(enabled)
    sensor = MultiLayerLidar(enabled.lidar3d)
    expected = np.vstack(
        [
            sensor.point_cloud(frame.measured_lidar3d, Pose2D(*result.estimated[index]))
            for index, frame in enumerate(ObservationStream(enabled).frames(), start=1)
        ]
    )
    assert len(expected) > 0
    np.testing.assert_allclose(result.map_points_3d, expected)
    assert 0.0 <= result.metrics["map3d_wall_f1_0p25"] <= 1.0
    target = export_result(result, tmp_path)
    exported = np.loadtxt(target / "map" / "point_cloud_3d.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(exported, expected)
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["lidar3d_point_count"] == len(expected)
    assert "estimated planar pose" in metadata["lidar3d_map_pose_source"]
    assert metadata["map3d_evaluation"]["reference"] == "static vertical wall surfaces only"

    plain = run_simulation(base)
    assert plain.map_points_3d.shape == (0, 3)
    plain_target = export_result(plain, tmp_path)
    assert not (plain_target / "map" / "point_cloud_3d.csv").exists()
