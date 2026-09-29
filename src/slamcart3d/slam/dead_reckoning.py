"""Dead-reckoning mapping baseline used to validate the end-to-end pipeline."""

from __future__ import annotations

import numpy as np

from slamcart3d.core.models import IMUSample, LidarScan, Pose2D
from slamcart3d.robots.differential_drive import DifferentialDrive
from slamcart3d.slam.base import SLAMAlgorithm


class DeadReckoningSLAM(SLAMAlgorithm):
    """Integrate wheel odometry and project LiDAR endpoints into a point map.

    This deliberately simple baseline does not close loops. It establishes a
    reproducible lower bound for later SLAM algorithms without claiming more.
    """

    def __init__(
        self,
        wheel_radius_m: float,
        wheel_base_m: float,
        lidar_max_range_m: float,
        map_stride: int = 5,
    ) -> None:
        self.wheel_radius_m = wheel_radius_m
        self.wheel_base_m = wheel_base_m
        self.lidar_max_range_m = lidar_max_range_m
        self.map_stride = map_stride
        self._robot: DifferentialDrive | None = None
        self._chunks: list[np.ndarray] = []
        self._updates = 0

    def initialize(self, pose: Pose2D) -> None:
        self._robot = DifferentialDrive(self.wheel_radius_m, self.wheel_base_m, pose)
        self._chunks = []
        self._updates = 0

    def update(
        self,
        timestamp: float,
        left_rate: float,
        right_rate: float,
        dt_s: float,
        scan: LidarScan,
        imu: IMUSample | None = None,
    ) -> Pose2D:
        del timestamp, imu
        if self._robot is None:
            raise RuntimeError("algorithm must be initialized before update")
        pose = self._robot.step(left_rate, right_rate, dt_s)
        ranges = scan.ranges[:: self.map_stride]
        angles = scan.angles[:: self.map_stride] + pose.yaw
        valid = np.isfinite(ranges) & (ranges > 0.02) & (ranges < 0.995 * self.lidar_max_range_m)
        if np.any(valid):
            points = np.column_stack(
                (
                    pose.x + ranges[valid] * np.cos(angles[valid]),
                    pose.y + ranges[valid] * np.sin(angles[valid]),
                )
            )
            self._chunks.append(points)
        self._updates += 1
        return pose

    def reset(self) -> None:
        self._robot = None
        self._chunks = []
        self._updates = 0

    def get_pose(self) -> Pose2D:
        if self._robot is None:
            raise RuntimeError("algorithm is not initialized")
        return self._robot.pose

    def get_map(self) -> np.ndarray:
        return np.vstack(self._chunks) if self._chunks else np.empty((0, 2), dtype=float)

    def get_diagnostics(self) -> dict[str, float | int | str]:
        return {"algorithm": "dead_reckoning", "updates": self._updates, "map_points": len(self.get_map())}
