import math

import numpy as np
import pytest

from slamcart3d.core.config import DegradationConfig, ExperimentConfig, MotionSegment, SimulationConfig
from slamcart3d.experiments.experiment import Experiment
from slamcart3d.simulation.motion import motion_command
from slamcart3d.simulation.observations import ObservationStream


def test_stop_turn_profile_alternates_translation_and_rotation() -> None:
    config = SimulationConfig(duration_s=0.8, timestep_s=0.1, motion_profile="stop_turn")
    commands = [motion_command(config, index, 8) for index in range(8)]
    assert commands == [(0.4, 0.0)] * 2 + [(0.0, 0.2)] * 2 + [(0.4, 0.0)] * 2 + [(0.0, 0.2)] * 2


def test_stop_turn_imu_reports_velocity_discontinuities() -> None:
    config = ExperimentConfig(
        degradation=DegradationConfig(lidar_noise_std_m=0.0, wheel_rate_noise_std_rad_s=0.0),
        simulation=SimulationConfig(duration_s=0.8, timestep_s=0.1, motion_profile="stop_turn"),
    )
    frames = list(ObservationStream(config).frames())
    accelerations = [frame.ideal_imu.longitudinal_acceleration_m_s2 for frame in frames]
    np.testing.assert_allclose(accelerations, [4.0, 0.0, -4.0, 0.0, 4.0, 0.0, -4.0, 0.0])
    assert frames[2].ideal_imu.yaw_rate_rad_s == pytest.approx(0.2)
    np.testing.assert_allclose(Experiment(config).run().estimated, Experiment(config).run().ground_truth)


def test_sinusoidal_yaw_profile_is_deterministic_and_nonconstant() -> None:
    config = SimulationConfig(duration_s=1.0, timestep_s=0.1, motion_profile="sinusoidal_yaw")
    rates = [motion_command(config, index, 10)[1] for index in range(10)]
    assert rates[0] == pytest.approx(config.yaw_rate_rad_s)
    assert rates[5] == pytest.approx(config.yaw_rate_rad_s * math.cos(math.pi * 0.1))
    assert np.ptp(rates) > 0.0


def test_invalid_motion_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported motion profile"):
        ExperimentConfig(simulation=SimulationConfig(motion_profile="unknown"))


def test_scripted_motion_replays_exact_piecewise_commands_and_ground_truth() -> None:
    segments = (
        MotionSegment(1.0, 0.4, 0.0),
        MotionSegment(1.0, 0.0, math.pi / 2.0),
        MotionSegment(1.0, 0.4, 0.0),
        MotionSegment(1.0, 0.0, -math.pi / 2.0),
    )
    config = ExperimentConfig(
        degradation=DegradationConfig(lidar_noise_std_m=0.0, wheel_rate_noise_std_rad_s=0.0),
        simulation=SimulationConfig(
            duration_s=4.0, timestep_s=0.1, motion_profile="scripted", motion_segments=segments
        ),
    )
    commands = [motion_command(config.simulation, index, 40) for index in range(40)]
    assert commands == [
        (segment.linear_velocity_m_s, segment.yaw_rate_rad_s) for segment in segments for _ in range(10)
    ]
    result = Experiment(config).run()
    np.testing.assert_allclose(result.ground_truth[-1], [5.4, 4.4, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.estimated, result.ground_truth, atol=1e-12)
    assert result.metrics["ate_rmse_m"] == pytest.approx(0.0, abs=1e-12)


def test_scripted_motion_round_trips_through_config_mapping() -> None:
    config = ExperimentConfig(
        simulation=SimulationConfig(
            duration_s=0.2,
            timestep_s=0.1,
            motion_profile="scripted",
            motion_segments=(MotionSegment(0.1, 0.5, 0.0), MotionSegment(0.1, 0.0, 0.5)),
        )
    )
    restored = ExperimentConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.simulation.motion_segments[1].yaw_rate_rad_s == 0.5


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ((), "exactly cover"),
        ((MotionSegment(0.1, 0.4, 0.0),), "exactly cover"),
        ((MotionSegment(0.15, 0.4, 0.0), MotionSegment(0.05, 0.0, 0.4)), "multiple"),
        ((MotionSegment(-0.1, 0.4, 0.0), MotionSegment(0.3, 0.0, 0.4)), "positive"),
    ],
)
def test_invalid_scripted_segments_are_rejected(segments, message) -> None:
    with pytest.raises(ValueError, match=message):
        ExperimentConfig(
            simulation=SimulationConfig(
                duration_s=0.2, timestep_s=0.1, motion_profile="scripted", motion_segments=segments
            )
        )


def test_scripted_motion_rejects_non_mapping_segments() -> None:
    with pytest.raises(ValueError, match="sequence of mappings"):
        ExperimentConfig.from_dict({"simulation": {"motion_segments": [0.1]}})
