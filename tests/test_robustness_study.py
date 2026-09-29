import csv
from dataclasses import replace

import numpy as np
import pytest

from slamcart3d.core.config import (
    ExperimentConfig,
    Lidar3DConfig,
    LidarConfig,
    MotionSegment,
    OccupancyGridConfig,
    SimulationConfig,
)
from slamcart3d.metrics.direction import failure_margin
from slamcart3d.robustness.study import (
    STUDY_DEGRADATION_PARAMETER_NAMES,
    SimulationMargin,
    export_frontier_study,
    run_frontier_study,
)


def test_real_simulation_frontier_runs_multiple_seeds(tmp_path) -> None:
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.3, timestep_s=0.1),
    )
    study = run_frontier_study(
        config,
        ("wheel_rate_noise_std_rad_s", "odometry_dropout_probability"),
        np.array([0.0, 0.0]),
        np.array([0.5, 0.4]),
        "ate_rmse_m",
        0.02,
        (1, 2),
        budget=8,
        grid_resolution=9,
    )
    assert len(study.records) == 8
    assert all(record["count"] == 2 for record in study.records)
    assert study.failure_probability.shape == (9, 9)
    assert study.failure_uncertainty.shape == (9, 9)
    np.testing.assert_allclose(study.success_probability + study.failure_probability, 1.0)
    destination = export_frontier_study(study, tmp_path)
    assert (destination / "config.yaml").exists()
    assert (destination / "sensitivity.csv").exists()
    assert len((destination / "seed_runs.csv").read_text(encoding="utf-8").splitlines()) == 17


def test_study_can_cross_motion_speed_with_imu_bias() -> None:
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.3, timestep_s=0.1),
    )
    study = run_frontier_study(
        config,
        ("simulation.linear_velocity_m_s", "imu_gyro_bias_rad_s"),
        np.array([0.2, 0.0]),
        np.array([0.8, 0.1]),
        "ate_rmse_m",
        0.01,
        (1, 2),
        budget=8,
        grid_resolution=5,
    )
    assert len(study.records) == 8
    assert all("simulation.linear_velocity_m_s" in record for record in study.records)
    assert study.success_probability.shape == (5, 5)


def test_motion_frequency_axis_requires_sinusoidal_profile() -> None:
    config = ExperimentConfig()
    with pytest.raises(ValueError, match="sinusoidal_yaw"):
        run_frontier_study(
            config,
            ("simulation.motion_frequency_hz", "lidar_noise_std_m"),
            np.array([0.05, 0.0]),
            np.array([0.2, 0.1]),
            "ate_rmse_m",
            0.02,
            (1,),
            budget=8,
            grid_resolution=5,
        )


def test_global_speed_axis_rejects_scripted_profile() -> None:
    config = ExperimentConfig(
        simulation=SimulationConfig(
            duration_s=0.2,
            timestep_s=0.1,
            motion_profile="scripted",
            motion_segments=(MotionSegment(0.2, 0.4, 0.0),),
        )
    )
    with pytest.raises(ValueError, match="do not modify scripted"):
        SimulationMargin(
            config,
            ("simulation.linear_velocity_m_s", "lidar_noise_std_m"),
            "ate_rmse_m",
            0.05,
            (1,),
        )


def test_occupancy_frontier_requires_enabled_grid() -> None:
    with pytest.raises(ValueError, match="occupancy_grid.enabled"):
        SimulationMargin(
            ExperimentConfig(),
            ("lidar_noise_std_m", "lidar_dropout_probability"),
            "occupancy_occupied_iou",
            0.2,
            (1,),
        )
    config = ExperimentConfig(occupancy_grid=OccupancyGridConfig(enabled=True))
    SimulationMargin(
        config,
        ("lidar_noise_std_m", "lidar_dropout_probability"),
        "occupancy_occupied_iou",
        0.2,
        (1,),
    )


def test_spatial_geometry_and_boolean_are_not_asrf_axes() -> None:
    assert "dropout_region_enabled" not in STUDY_DEGRADATION_PARAMETER_NAMES
    assert "dropout_region_x_min_m" not in STUDY_DEGRADATION_PARAMETER_NAMES
    assert "camera_rgb_noise_std" not in STUDY_DEGRADATION_PARAMETER_NAMES
    with pytest.raises(ValueError, match="unknown study parameters"):
        SimulationMargin(
            ExperimentConfig(),
            ("dropout_region_enabled", "lidar_noise_std_m"),
            "ate_rmse_m",
            0.05,
            (0,),
        )


def test_asrf_accepts_higher_is_better_3d_wall_f1() -> None:
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        lidar3d=Lidar3DConfig(enabled=True, azimuth_beams=3, elevation_beams=3),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )
    study = run_frontier_study(
        config,
        ("lidar_noise_std_m", "lidar_dropout_probability"),
        np.array([0.0, 0.0]),
        np.array([0.1, 0.5]),
        "map3d_wall_f1_0p25",
        0.5,
        (0,),
        budget=8,
        grid_resolution=5,
    )
    assert len(study.records) == 8
    for record in study.records:
        assert record["failed"] == (record["median"] <= 0.5)
    assert failure_margin(0.4, "map3d_wall_f1_0p25", 0.5) > 0.0


def test_seed_run_failure_respects_higher_is_better_metric(tmp_path) -> None:
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )
    original = run_frontier_study(
        config,
        ("wheel_rate_noise_std_rad_s", "odometry_dropout_probability"),
        np.array([0.0, 0.0]),
        np.array([0.1, 0.1]),
        "ate_rmse_m",
        0.02,
        (1, 2),
        budget=8,
        grid_resolution=5,
    )
    record = {**original.records[0], "metric": "tracking_availability", "seed_values": [0.8, 0.2]}
    study = replace(
        original,
        records=(record,),
        metric_name="tracking_availability",
        failure_threshold=0.5,
    )
    destination = export_frontier_study(study, tmp_path)
    with (destination / "seed_runs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["failed"] for row in rows] == ["False", "True"]
