"""Perspective synthetic RGB-D frames and independent camera degradations."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from slamcart3d.core.config import (
    CameraConfig,
    DegradationConfig,
    ExperimentConfig,
    Lidar3DConfig,
    LidarConfig,
    SimulationConfig,
)
from slamcart3d.core.models import Pose2D
from slamcart3d.datasets.generator import generate_dataset
from slamcart3d.degradation.engine import DegradationEngine
from slamcart3d.sensors.camera import SyntheticCamera
from slamcart3d.simulation.observations import ObservationStream
from slamcart3d.simulation.runner import run_simulation
from slamcart3d.worlds.geometry import WallSegment, World3D


def test_camera_center_ray_and_ground_depth_are_geometric() -> None:
    world = World3D("test", (WallSegment(5.0, -2.0, 5.0, 2.0, height_m=2.0),))
    camera = SyntheticCamera(CameraConfig(enabled=True, width_px=3, height_px=3))
    frame = camera.capture(0.0, Pose2D(0.0, 0.0, 0.0), world)
    assert frame.rgb.shape == (3, 3, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.depth_m.shape == (3, 3)
    assert frame.depth_m[1, 1] == pytest.approx(5.0)
    assert np.isnan(frame.depth_m[0, 1])
    assert 0.0 < frame.depth_m[2, 1] < 5.0


def test_camera_degradation_is_reproducible_and_preserves_ideal_frame() -> None:
    world = World3D("test", (WallSegment(5.0, -2.0, 5.0, 2.0, height_m=2.0),))
    frame = SyntheticCamera(CameraConfig(enabled=True, width_px=8, height_px=6)).capture(
        0.1, Pose2D(0.0, 0.0, 0.0), world
    )
    original_rgb = frame.rgb.copy()
    original_depth = frame.depth_m.copy()
    clean = DegradationEngine(DegradationConfig(), 3).degrade_camera(frame)
    assert clean is not None
    np.testing.assert_array_equal(clean.rgb, frame.rgb)
    np.testing.assert_allclose(clean.depth_m, frame.depth_m, equal_nan=True)

    config = DegradationConfig(
        camera_rgb_noise_std=8.0,
        camera_depth_noise_std_m=0.05,
        camera_blur_radius_px=1,
        camera_lighting_factor=0.6,
        camera_texture_reduction_fraction=0.2,
        camera_occlusion_fraction=0.25,
    )
    first = DegradationEngine(config, 9).degrade_camera(frame)
    second = DegradationEngine(config, 9).degrade_camera(frame)
    assert first is not None and second is not None
    np.testing.assert_array_equal(first.rgb, second.rgb)
    np.testing.assert_allclose(first.depth_m, second.depth_m, equal_nan=True)
    assert not np.array_equal(first.rgb, frame.rgb)
    assert np.isnan(first.depth_m).sum() > np.isnan(frame.depth_m).sum()
    np.testing.assert_array_equal(frame.rgb, original_rgb)
    np.testing.assert_allclose(frame.depth_m, original_depth, equal_nan=True)


def test_camera_frame_dropout_combines_time_and_spatial_region() -> None:
    frame = SyntheticCamera(CameraConfig(enabled=True, width_px=3, height_px=3)).capture(
        0.0,
        Pose2D(0.0, 0.0, 0.0),
        World3D("test", (WallSegment(5.0, -2.0, 5.0, 2.0),)),
    )
    engine = DegradationEngine(
        DegradationConfig(
            camera_frame_dropout_probability=1.0,
            dropout_time_profile="linear_ramp",
            dropout_region_enabled=True,
            dropout_region_x_min_m=1.0,
            dropout_region_x_max_m=2.0,
            dropout_region_y_min_m=1.0,
            dropout_region_y_max_m=2.0,
        ),
        5,
    )
    inside = Pose2D(1.5, 1.5, 0.0)
    outside = Pose2D(0.0, 0.0, 0.0)
    assert engine.degrade_camera(frame, progress=0.0, pose=inside) is not None
    assert engine.degrade_camera(frame, progress=1.0, pose=outside) is not None
    assert engine.degrade_camera(frame, progress=1.0, pose=inside) is None


def test_optional_camera_preserves_other_sensor_streams_and_exports_images(tmp_path) -> None:
    base = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        lidar3d=Lidar3DConfig(enabled=True, azimuth_beams=3, elevation_beams=3),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )
    enabled = replace(base, camera=CameraConfig(enabled=True, width_px=8, height_px=6))
    plain_frames = list(ObservationStream(base).frames())
    camera_frames = list(ObservationStream(enabled).frames())
    for plain, enhanced in zip(plain_frames, camera_frames, strict=True):
        assert plain.ideal_camera is None
        assert enhanced.ideal_camera is not None
        assert enhanced.measured_camera is not None
        np.testing.assert_allclose(
            plain.measured_lidar.ranges, enhanced.measured_lidar.ranges, equal_nan=True
        )
        np.testing.assert_allclose(
            plain.measured_lidar3d.ranges, enhanced.measured_lidar3d.ranges, equal_nan=True
        )
        assert plain.measured_left_rate_rad_s == enhanced.measured_left_rate_rad_s
        assert plain.measured_imu.yaw_rate_rad_s == enhanced.measured_imu.yaw_rate_rad_s
    target = generate_dataset(enabled, tmp_path)
    with (target / "camera_index.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    for row in rows:
        rgb = target / row["measured_rgb_png"]
        depth = target / row["measured_depth_npy"]
        assert rgb.is_file() and depth.is_file()
        with Image.open(rgb) as image:
            assert image.size == (8, 6)
        assert np.load(depth).shape == (6, 8)
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["camera_enabled"]
    assert metadata["camera_resolution_px"] == [8, 6]
    plain_run = run_simulation(base)
    camera_run = run_simulation(enabled)
    np.testing.assert_allclose(plain_run.ground_truth, camera_run.ground_truth)
    np.testing.assert_allclose(plain_run.estimated, camera_run.estimated)
    np.testing.assert_allclose(plain_run.map_points, camera_run.map_points)
    for metric, value in plain_run.metrics.items():
        if metric != "runtime_s":
            assert camera_run.metrics[metric] == pytest.approx(value)


def test_dropped_camera_frame_has_no_measured_image_paths(tmp_path) -> None:
    config = ExperimentConfig(
        camera=CameraConfig(enabled=True, width_px=4, height_px=3),
        lidar=LidarConfig(beam_count=8),
        degradation=DegradationConfig(camera_frame_dropout_probability=1.0),
        simulation=SimulationConfig(duration_s=0.1, timestep_s=0.1),
    )
    target = generate_dataset(config, tmp_path)
    with (target / "camera_index.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert (target / row["ideal_rgb_png"]).is_file()
    assert row["measured_rgb_png"] == ""
    assert row["measured_depth_npy"] == ""


def test_camera_configuration_validation() -> None:
    with pytest.raises(ValueError, match="at least two pixels"):
        ExperimentConfig(camera=CameraConfig(enabled=True, width_px=1))
    with pytest.raises(ValueError, match="camera_blur_radius_px"):
        ExperimentConfig(degradation=DegradationConfig(camera_blur_radius_px=6))
    with pytest.raises(ValueError, match="probabilities"):
        ExperimentConfig(degradation=DegradationConfig(camera_frame_dropout_probability=1.1))
    with pytest.raises(ValueError, match="at most 160 x 120"):
        ExperimentConfig(camera=CameraConfig(enabled=True, width_px=161))
