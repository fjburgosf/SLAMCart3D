import numpy as np

from slamcart3d.core.models import Pose2D
from slamcart3d.robots.differential_drive import DifferentialDrive


def test_straight_line_kinematics() -> None:
    robot = DifferentialDrive(0.1, 0.5, Pose2D(0.0, 0.0, 0.0))
    pose = robot.step(2.0, 2.0, 1.5)
    np.testing.assert_allclose(pose.as_array(), [0.3, 0.0, 0.0], atol=1e-12)


def test_constant_turn_returns_expected_arc() -> None:
    robot = DifferentialDrive(0.1, 0.5, Pose2D(0.0, 0.0, 0.0))
    pose = robot.step(1.0, 3.0, 1.0)
    expected_radius = 0.2 / 0.4
    np.testing.assert_allclose(
        pose.as_array(),
        [expected_radius * np.sin(0.4), expected_radius * (1.0 - np.cos(0.4)), 0.4],
        atol=1e-12,
    )
