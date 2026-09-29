from dataclasses import replace

import numpy as np
import pytest

from slamcart3d.core.config import (
    DegradationConfig,
    ExperimentConfig,
    Lidar3DConfig,
    LidarConfig,
    SimulationConfig,
)
from slamcart3d.core.models import IMUSample, LidarScan, Pose2D
from slamcart3d.degradation.engine import DegradationEngine
from slamcart3d.simulation.observations import ObservationStream


def test_degradation_is_reproducible_and_preserves_input() -> None:
    config = DegradationConfig(lidar_noise_std_m=0.2, lidar_dropout_probability=0.2)
    original = LidarScan(0.0, np.linspace(-1.0, 1.0, 20), np.full(20, 4.0))
    first = DegradationEngine(config, 9).degrade_lidar(original, 10.0)
    second = DegradationEngine(config, 9).degrade_lidar(original, 10.0)
    np.testing.assert_allclose(first.ranges, second.ranges, equal_nan=True)
    np.testing.assert_array_equal(original.ranges, np.full(20, 4.0))


def test_lidar_range_reduction_marks_unreachable_returns_missing() -> None:
    config = DegradationConfig(lidar_noise_std_m=0.0, lidar_range_reduction_fraction=0.5)
    original = LidarScan(0.0, np.array([0.0, 0.5]), np.array([3.0, 7.0]))
    measured = DegradationEngine(config, 1).degrade_lidar(original, 10.0)
    assert measured.ranges[0] == 3.0
    assert np.isnan(measured.ranges[1])
    np.testing.assert_array_equal(original.ranges, [3.0, 7.0])


def test_lidar_sector_dropout_uses_robot_frame_angles() -> None:
    config = DegradationConfig(
        lidar_noise_std_m=0.0,
        lidar_sector_dropout_center_deg=0.0,
        lidar_sector_dropout_width_deg=90.0,
    )
    original = LidarScan(0.0, np.deg2rad([-90.0, -45.0, 0.0, 45.0, 90.0]), np.ones(5))
    measured = DegradationEngine(config, 1).degrade_lidar(original, 10.0)
    np.testing.assert_array_equal(np.isnan(measured.ranges), [False, True, True, True, False])


def test_reduced_range_also_limits_false_returns() -> None:
    config = DegradationConfig(
        lidar_noise_std_m=0.0,
        lidar_outlier_probability=1.0,
        lidar_range_reduction_fraction=0.9,
    )
    original = LidarScan(0.0, np.linspace(-1.0, 1.0, 100), np.full(100, 0.5))
    measured = DegradationEngine(config, 1).degrade_lidar(original, 10.0)
    assert np.all(np.isnan(measured.ranges) | (measured.ranges <= 1.0))


def test_no_hit_ideal_beams_remain_missing_after_noise() -> None:
    config = DegradationConfig(lidar_noise_std_m=0.2)
    original = LidarScan(0.0, np.array([-0.5, 0.5]), np.array([10.0, 3.0]))
    measured = DegradationEngine(config, 1).degrade_lidar(original, 10.0)
    assert np.isnan(measured.ranges[0])
    assert np.isfinite(measured.ranges[1])
    assert original.ranges[0] == 10.0


def test_dropout_linear_ramp_reaches_configured_severity() -> None:
    engine = DegradationEngine(
        DegradationConfig(
            wheel_rate_noise_std_rad_s=0.0,
            odometry_dropout_probability=1.0,
            dropout_time_profile="linear_ramp",
        ),
        3,
    )
    assert engine.degrade_wheel_rates(1.0, 2.0, progress=0.0) == (1.0, 2.0)
    assert engine.degrade_wheel_rates(1.0, 2.0, progress=1.0) == (0.0, 0.0)


def test_abrupt_dropout_changes_observations_not_ground_truth() -> None:
    base = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.4, timestep_s=0.1),
        degradation=DegradationConfig(
            lidar_noise_std_m=0.0,
            wheel_rate_noise_std_rad_s=0.0,
            lidar_dropout_probability=1.0,
            odometry_dropout_probability=1.0,
            imu_dropout_probability=1.0,
            dropout_time_profile="abrupt",
            dropout_transition_fraction=0.5,
        ),
    )
    frames = list(ObservationStream(base).frames())
    assert np.isfinite(frames[0].measured_lidar.ranges).any()
    assert np.all(np.isnan(frames[1].measured_lidar.ranges))
    assert frames[0].measured_left_rate_rad_s != 0.0
    assert frames[1].measured_left_rate_rad_s == 0.0
    assert np.isfinite(frames[0].measured_imu.yaw_rate_rad_s)
    assert np.isnan(frames[1].measured_imu.yaw_rate_rad_s)
    reference = list(
        ObservationStream(
            ExperimentConfig(lidar=base.lidar, simulation=base.simulation, seed=base.seed)
        ).frames()
    )
    for degraded, clean in zip(frames, reference, strict=True):
        np.testing.assert_allclose(degraded.ground_truth.as_array(), clean.ground_truth.as_array())


def test_spatial_dropout_is_bounded_and_requires_pose() -> None:
    config = DegradationConfig(
        lidar_noise_std_m=0.0,
        wheel_rate_noise_std_rad_s=0.0,
        lidar_dropout_probability=1.0,
        odometry_dropout_probability=1.0,
        imu_dropout_probability=1.0,
        dropout_region_enabled=True,
        dropout_region_x_min_m=1.0,
        dropout_region_x_max_m=2.0,
        dropout_region_y_min_m=3.0,
        dropout_region_y_max_m=4.0,
    )
    scan = LidarScan(0.0, np.array([-0.5, 0.5]), np.array([2.0, 3.0]))
    outside = Pose2D(0.0, 3.5, 0.0)
    boundary = Pose2D(1.0, 4.0, 0.0)
    engine = DegradationEngine(config, 11)
    np.testing.assert_allclose(engine.degrade_lidar(scan, 10.0, pose=outside).ranges, scan.ranges)
    assert np.isnan(engine.degrade_lidar(scan, 10.0, pose=boundary).ranges).all()
    assert engine.degrade_wheel_rates(1.0, 2.0, pose=outside) == (1.0, 2.0)
    assert engine.degrade_wheel_rates(1.0, 2.0, pose=boundary) == (0.0, 0.0)
    sample = IMUSample(0.0, 0.2, 0.3)
    assert np.isfinite(engine.degrade_imu(sample, 0.1, ExperimentConfig().imu, pose=outside).yaw_rate_rad_s)
    assert np.isnan(engine.degrade_imu(sample, 0.1, ExperimentConfig().imu, pose=boundary).yaw_rate_rad_s)
    with pytest.raises(ValueError, match="requires a robot pose"):
        engine.degrade_lidar(scan, 10.0)


def test_spatial_and_temporal_dropout_compose_without_changing_truth() -> None:
    degradation = DegradationConfig(
        lidar_noise_std_m=0.0,
        wheel_rate_noise_std_rad_s=0.0,
        lidar_dropout_probability=1.0,
        odometry_dropout_probability=1.0,
        imu_dropout_probability=1.0,
        dropout_time_profile="abrupt",
        dropout_transition_fraction=0.4,
        dropout_region_enabled=True,
        dropout_region_x_min_m=5.35,
        dropout_region_x_max_m=5.65,
        dropout_region_y_min_m=3.0,
        dropout_region_y_max_m=5.0,
    )
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        lidar3d=Lidar3DConfig(enabled=True, azimuth_beams=3, elevation_beams=3),
        simulation=SimulationConfig(
            duration_s=1.0,
            timestep_s=0.1,
            linear_velocity_m_s=1.0,
            yaw_rate_rad_s=0.0,
        ),
        degradation=degradation,
    )
    frames = list(ObservationStream(config).frames())
    repeated = list(ObservationStream(config).frames())
    reference = list(ObservationStream(replace(config, degradation=DegradationConfig())).frames())
    assert len(frames) == 10
    for frame, replay, clean in zip(frames, repeated, reference, strict=True):
        np.testing.assert_allclose(frame.ground_truth.as_array(), clean.ground_truth.as_array())
        np.testing.assert_allclose(frame.measured_lidar.ranges, replay.measured_lidar.ranges, equal_nan=True)
        in_region = 5.35 <= frame.ground_truth.x <= 5.65
        if in_region:
            assert np.isnan(frame.measured_lidar.ranges).all()
            assert np.isnan(frame.measured_lidar3d.ranges).all()
            assert frame.measured_left_rate_rad_s == 0.0
            assert np.isnan(frame.measured_imu.yaw_rate_rad_s)
        else:
            assert np.isfinite(frame.measured_lidar.ranges).any()
            assert frame.measured_left_rate_rad_s != 0.0
            assert np.isfinite(frame.measured_imu.yaw_rate_rad_s)


def test_spatial_dropout_region_config_validation_and_roundtrip() -> None:
    config = ExperimentConfig(
        degradation=DegradationConfig(
            dropout_region_enabled=True,
            dropout_region_x_min_m=-1.0,
            dropout_region_x_max_m=2.0,
            dropout_region_y_min_m=3.0,
            dropout_region_y_max_m=4.0,
        )
    )
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="positive x and y spans"):
        replace(
            config,
            degradation=replace(config.degradation, dropout_region_x_max_m=-1.0),
        )
