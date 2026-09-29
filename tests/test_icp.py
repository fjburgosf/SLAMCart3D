import numpy as np

from slamcart3d.core.config import (
    DegradationConfig,
    ExperimentConfig,
    LidarConfig,
    SimulationConfig,
    SLAMConfig,
)
from slamcart3d.experiments.experiment import Experiment
from slamcart3d.slam.icp_odometry import point_to_line_icp


def test_point_to_line_icp_recovers_motion_with_prior() -> None:
    horizontal = np.column_stack((np.linspace(0.0, 1.0, 12), np.zeros(12)))
    vertical = np.column_stack((np.ones(12), np.linspace(0.0, 1.0, 12)))
    target = np.vstack((horizontal, vertical[1:]))
    angle = 0.04
    expected_rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    expected_translation = np.array([0.08, -0.04])
    source = (target - expected_translation) @ expected_rotation
    initial_translation = np.array([0.07, -0.03])
    rotation, translation, error, inliers = point_to_line_icp(
        source,
        target,
        expected_rotation,
        initial_translation,
        max_correspondence_m=0.5,
    )
    assert inliers >= 18
    assert error < 0.02
    assert np.linalg.norm(translation - expected_translation) < np.linalg.norm(
        initial_translation - expected_translation
    )
    assert abs(np.arctan2(rotation[1, 0], rotation[0, 0]) - angle) < 0.01


def test_icp_trajectory_does_not_drift_on_ideal_measurements() -> None:
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=90),
        degradation=DegradationConfig(lidar_noise_std_m=0.0, wheel_rate_noise_std_rad_s=0.0),
        simulation=SimulationConfig(duration_s=4.0, timestep_s=0.1),
        slam=SLAMConfig(algorithm="icp_odometry"),
    )
    result = Experiment(config).run()
    assert result.metrics["ate_rmse_m"] < 0.03
    assert result.metrics["final_position_error_m"] < 0.08
    assert result.diagnostics["fallbacks"] == 0
    assert result.metrics["icp_registration_availability"] == 1.0


def test_icp_registration_availability_detects_blind_lidar() -> None:
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=12),
        degradation=DegradationConfig(lidar_dropout_probability=1.0),
        simulation=SimulationConfig(duration_s=0.4, timestep_s=0.1),
        slam=SLAMConfig(algorithm="icp_odometry"),
    )
    result = Experiment(config).run()
    assert result.metrics["icp_registration_availability"] == 0.0
    assert result.metrics["tracking_availability"] == 1.0
    assert result.metrics["map_recall_0p25"] == 0.0
