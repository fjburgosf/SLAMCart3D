import pytest

from slamcart3d.core.config import DegradationConfig, ExperimentConfig, Lidar3DConfig, SimulationConfig


def test_non_integral_simulation_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        ExperimentConfig(simulation=SimulationConfig(duration_s=1.05, timestep_s=0.1))


def test_negative_noise_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ExperimentConfig(degradation=DegradationConfig(lidar_noise_std_m=-0.1))


def test_invalid_lidar_sector_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="sector dropout width"):
        ExperimentConfig(degradation=DegradationConfig(lidar_sector_dropout_width_deg=400.0))


def test_invalid_dropout_time_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="dropout_time_profile"):
        ExperimentConfig(degradation=DegradationConfig(dropout_time_profile="unknown"))
    with pytest.raises(ValueError, match="dropout_transition_fraction"):
        ExperimentConfig(degradation=DegradationConfig(dropout_transition_fraction=1.1))


def test_invalid_3d_lidar_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="3D LiDAR requires"):
        ExperimentConfig(lidar3d=Lidar3DConfig(enabled=True, elevation_beams=1))
