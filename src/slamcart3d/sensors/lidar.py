"""Planar LiDAR simulation against the world geometry."""

from __future__ import annotations

import numpy as np

from slamcart3d.core.models import LidarScan, Pose2D
from slamcart3d.worlds.geometry import World3D


class PlanarLidar:
    """Deterministic ideal LiDAR; degradations are applied by a separate engine."""

    def __init__(self, beam_count: int, field_of_view_deg: float, max_range_m: float) -> None:
        self.max_range_m = max_range_m
        half_angle = np.deg2rad(field_of_view_deg) / 2.0
        self.angles = np.linspace(-half_angle, half_angle, beam_count, dtype=float)

    def scan(self, timestamp: float, pose: Pose2D, world: World3D) -> LidarScan:
        """Cast all beams from *pose* and return ideal ranges."""

        ranges = np.array(
            [
                world.ray_distance(pose.x, pose.y, pose.yaw + angle, self.max_range_m, timestamp)
                for angle in self.angles
            ],
            dtype=float,
        )
        return LidarScan(timestamp=timestamp, angles=self.angles.copy(), ranges=ranges)
