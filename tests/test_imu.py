import csv
from dataclasses import replace

import numpy as np
import pytest

from slamcart3d.core.config import (
    DegradationConfig,
    ExperimentConfig,
    IMUConfig,
    SimulationConfig,
    SLAMConfig,
)
from slamcart3d.core.models import LidarScan, Pose2D
from slamcart3d.datasets.generator import generate_dataset
from slamcart3d.experiments.experiment import Experiment
from slamcart3d.simulation.observations import ObservationStream
from slamcart3d.slam.dead_reckoning import DeadReckoningSLAM
from slamcart3d.slam.wheel_imu_fusion import WheelIMUFusionSLAM


def _short_config(**degradation_values: float) -> ExperimentConfig:
    return ExperimentConfig(
        degradation=DegradationConfig(
            lidar_noise_std_m=0.0,
            wheel_rate_noise_std_rad_s=0.0,
            **degradation_values,
        ),
        simulation=SimulationConfig(duration_s=1.0, timestep_s=0.1),
    )


def test_ideal_imu_matches_planar_motion_and_replays() -> None:
    config = _short_config(imu_gyro_noise_std_rad_s=0.03, imu_gyro_random_walk_std_rad_s_sqrt_s=0.01)
    stream = ObservationStream(config)
    first = list(stream.frames())
    second = list(stream.frames())
    assert first[0].ideal_imu is not None
    assert first[0].ideal_imu.yaw_rate_rad_s == pytest.approx(0.2)
    assert first[0].ideal_imu.longitudinal_acceleration_m_s2 == pytest.approx(4.0)
    assert first[0].ideal_imu.roll_rate_rad_s == 0.0
    assert first[0].ideal_imu.pitch_rate_rad_s == 0.0
    assert first[0].ideal_imu.lateral_acceleration_m_s2 == pytest.approx(0.08)
    assert first[0].ideal_imu.vertical_specific_force_m_s2 == pytest.approx(9.80665)
    assert first[1].ideal_imu.longitudinal_acceleration_m_s2 == pytest.approx(0.0)
    for one, two in zip(first, second, strict=True):
        assert one.measured_imu == two.measured_imu
        assert one.ground_truth == two.ground_truth


def test_imu_rng_does_not_change_legacy_lidar_or_wheel_stream() -> None:
    enabled = _short_config(imu_gyro_noise_std_rad_s=0.1)
    disabled = replace(enabled, imu=IMUConfig(enabled=False))
    for one, two in zip(
        ObservationStream(enabled).frames(), ObservationStream(disabled).frames(), strict=True
    ):
        np.testing.assert_allclose(one.measured_lidar.ranges, two.measured_lidar.ranges, equal_nan=True)
        assert one.measured_left_rate_rad_s == two.measured_left_rate_rad_s
        assert one.measured_right_rate_rad_s == two.measured_right_rate_rad_s
        assert two.ideal_imu is None and two.measured_imu is None


def test_imu_bias_and_dropout_affect_fusion_without_changing_truth() -> None:
    base = _short_config()
    fused = replace(base, slam=SLAMConfig(algorithm="wheel_imu_fusion"))
    biased = replace(fused, degradation=replace(fused.degradation, imu_gyro_bias_rad_s=0.1))
    missing = replace(fused, degradation=replace(fused.degradation, imu_dropout_probability=1.0))
    clean_result = Experiment(fused).run()
    biased_result = Experiment(biased).run()
    missing_result = Experiment(missing).run()
    np.testing.assert_allclose(clean_result.ground_truth, biased_result.ground_truth)
    np.testing.assert_allclose(clean_result.ground_truth, missing_result.ground_truth)
    assert clean_result.estimated[-1, 2] == pytest.approx(0.2)
    assert biased_result.estimated[-1, 2] == pytest.approx(0.3)
    np.testing.assert_allclose(missing_result.estimated, clean_result.ground_truth, atol=1e-12)
    assert missing_result.diagnostics["wheel_yaw_fallbacks"] == 10
    assert biased_result.diagnostics["imu_updates"] == 10


def test_imu_saturation_and_dataset_missing_fields(tmp_path) -> None:
    config = replace(
        _short_config(imu_gyro_bias_rad_s=2.0, imu_dropout_probability=1.0),
        imu=IMUConfig(gyro_saturation_rad_s=0.5),
    )
    destination = generate_dataset(config, tmp_path)
    with (destination / "imu.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 10
    assert all(row["measured_yaw_rate_rad_s"] == "" for row in rows)
    assert all(row["measured_roll_rate_rad_s"] == "" for row in rows)
    assert all(row["measured_vertical_specific_force_m_s2"] == "" for row in rows)
    assert float(rows[0]["ideal_yaw_rate_rad_s"]) == pytest.approx(0.2)
    assert float(rows[0]["ideal_lateral_acceleration_m_s2"]) == pytest.approx(0.08)
    assert float(rows[0]["ideal_vertical_specific_force_m_s2"]) == pytest.approx(9.80665)
    saturated = replace(config, degradation=replace(config.degradation, imu_dropout_probability=0.0))
    frame = next(ObservationStream(saturated).frames())
    assert frame.measured_imu.yaw_rate_rad_s == pytest.approx(0.5)


def test_auxiliary_imu_axes_receive_reproducible_noise_without_changing_truth() -> None:
    config = _short_config(
        imu_gyro_noise_std_rad_s=0.02,
        imu_accel_noise_std_m_s2=0.03,
        imu_gyro_random_walk_std_rad_s_sqrt_s=0.01,
    )
    first = next(ObservationStream(config).frames())
    second = next(ObservationStream(config).frames())
    assert first.measured_imu == second.measured_imu
    assert first.ideal_imu == second.ideal_imu
    assert first.measured_imu is not None and first.ideal_imu is not None
    assert first.measured_imu.roll_rate_rad_s != first.ideal_imu.roll_rate_rad_s
    assert first.measured_imu.lateral_acceleration_m_s2 != first.ideal_imu.lateral_acceleration_m_s2


def test_fusion_requires_enabled_imu() -> None:
    with pytest.raises(ValueError, match="enabled IMU"):
        ExperimentConfig(imu=IMUConfig(enabled=False), slam=SLAMConfig(algorithm="wheel_imu_fusion"))


@pytest.mark.parametrize("algorithm_class", [DeadReckoningSLAM, WheelIMUFusionSLAM])
def test_no_hit_lidar_ranges_are_not_added_to_point_map(algorithm_class) -> None:
    algorithm = algorithm_class(0.1, 0.5, 12.0)
    algorithm.initialize(Pose2D(5.0, 4.0, 0.0))
    scan = LidarScan(0.1, np.linspace(-1.0, 1.0, 10), np.full(10, 12.0))
    algorithm.update(0.1, 0.0, 0.0, 0.1, scan)
    assert algorithm.get_map().shape == (0, 2)
