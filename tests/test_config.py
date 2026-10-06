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


@pytest.mark.parametrize(
    "data",
    [
        {"seed": 1.7},
        {"seed": True},
        {"seed": "42"},
        {"name": "   "},
        {"name": 17},
        {"lidar": {"beam_count": 2.5}},
        {"lidar": {"beam_count": True}},
        {"lidar": {"max_range_m": "12"}},
        {"lidar3d": {"enabled": "false"}},
        {"camera": {"enabled": 1}},
        {"imu": {"enabled": "false"}},
        {"simulation": {"stop_at_walls": "false"}},
        {"simulation": {"duration_s": True}},
        {"slam": {"icp_max_iterations": 2.5}},
        {"slam": {"hybrid_use_imu": "false"}},
        {"robustness_score": {"enabled": "true"}},
        {"occupancy_grid": {"beam_stride": True}},
        {"simulation": None},
        {"lidar": []},
        {"sead": 2},
    ],
)
def test_ambiguous_configuration_is_rejected_without_coercion(data) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(data)


def test_configuration_wrapper_rejects_ignored_fields(tmp_path) -> None:
    target = tmp_path / "unexpected.yaml"
    target.write_text("experiment: {}\nseed: 19\n", encoding="utf-8")
    with pytest.raises(ValueError, match="wrapper"):
        ExperimentConfig.load(target)


def test_integer_physical_values_and_roundtrip_remain_valid() -> None:
    config = ExperimentConfig.from_dict({"robot": {"wheel_base_m": 1}, "seed": 7})
    assert ExperimentConfig.from_dict(config.to_dict()) == config
