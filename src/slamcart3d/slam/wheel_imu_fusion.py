"""Wheel translation and planar gyroscope heading odometry baseline."""

from __future__ import annotations

import numpy as np

from slamcart3d.core.models import IMUSample, LidarScan, Pose2D
from slamcart3d.robots.differential_drive import DifferentialDrive
from slamcart3d.slam.base import SLAMAlgorithm


class WheelIMUFusionSLAM(SLAMAlgorithm):
    """Use wheel translation and IMU yaw rate, falling back to wheel yaw.

    This is a deterministic sensor-fusion odometry baseline, not an optimal
    filter or a loop-closing SLAM algorithm. The accelerometer is recorded but
    not fused because a stationary bias can cause unbounded double-integration
    error without an independent velocity observation.
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
        self._imu_updates = 0
        self._wheel_yaw_fallbacks = 0

    def initialize(self, pose: Pose2D) -> None:
        self._robot = DifferentialDrive(self.wheel_radius_m, self.wheel_base_m, pose)
        self._chunks = []
        self._updates = 0
        self._imu_updates = 0
        self._wheel_yaw_fallbacks = 0

    def update(
        self,
        timestamp: float,
        left_rate: float,
        right_rate: float,
        dt_s: float,
        scan: LidarScan,
        imu: IMUSample | None = None,
    ) -> Pose2D:
        del timestamp
        if self._robot is None:
            raise RuntimeError("algorithm must be initialized before update")
        velocity = self.wheel_radius_m * (left_rate + right_rate) / 2.0
        if imu is not None and np.isfinite(imu.yaw_rate_rad_s):
            yaw_rate = imu.yaw_rate_rad_s
            self._imu_updates += 1
        else:
            yaw_rate = self.wheel_radius_m * (right_rate - left_rate) / self.wheel_base_m
            self._wheel_yaw_fallbacks += 1
        fused_left, fused_right = self._robot.wheel_rates(velocity, yaw_rate)
        pose = self._robot.step(fused_left, fused_right, dt_s)
        ranges = scan.ranges[:: self.map_stride]
        angles = scan.angles[:: self.map_stride] + pose.yaw
        valid = np.isfinite(ranges) & (ranges > 0.02) & (ranges < 0.995 * self.lidar_max_range_m)
        if np.any(valid):
            self._chunks.append(
                np.column_stack(
                    (
                        pose.x + ranges[valid] * np.cos(angles[valid]),
                        pose.y + ranges[valid] * np.sin(angles[valid]),
                    )
                )
            )
        self._updates += 1
        return pose

    def reset(self) -> None:
        self._robot = None
        self._chunks = []
        self._updates = 0
        self._imu_updates = 0
        self._wheel_yaw_fallbacks = 0

    def get_pose(self) -> Pose2D:
        if self._robot is None:
            raise RuntimeError("algorithm is not initialized")
        return self._robot.pose

    def get_map(self) -> np.ndarray:
        return np.vstack(self._chunks) if self._chunks else np.empty((0, 2), dtype=float)

    def get_diagnostics(self) -> dict[str, float | int | str]:
        return {
            "algorithm": "wheel_imu_fusion",
            "updates": self._updates,
            "imu_updates": self._imu_updates,
            "wheel_yaw_fallbacks": self._wheel_yaw_fallbacks,
            "map_points": len(self.get_map()),
        }
