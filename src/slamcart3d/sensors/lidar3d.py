"""Optional multilayer LiDAR over finite-height walls and a flat ground plane."""

from __future__ import annotations

import numpy as np

from slamcart3d.core.config import Lidar3DConfig
from slamcart3d.core.models import LidarScan3D, Pose2D
from slamcart3d.worlds.geometry import World3D


class MultiLayerLidar:
    """Generate reproducible slant ranges and global 3D points without rendering."""

    def __init__(self, config: Lidar3DConfig) -> None:
        self.config = config
        azimuths = np.linspace(
            -np.deg2rad(config.horizontal_fov_deg) / 2.0,
            np.deg2rad(config.horizontal_fov_deg) / 2.0,
            config.azimuth_beams,
        )
        elevations = np.linspace(
            -np.deg2rad(config.vertical_fov_deg) / 2.0,
            np.deg2rad(config.vertical_fov_deg) / 2.0,
            config.elevation_beams,
        )
        self.azimuths = np.tile(azimuths, config.elevation_beams)
        self.elevations = np.repeat(elevations, config.azimuth_beams)

    def scan(self, timestamp: float, pose: Pose2D, world: World3D) -> LidarScan3D:
        """Cast all configured rays; max range denotes no surface hit."""

        ranges = np.array(
            [
                world.ray_distance_3d(
                    pose.x,
                    pose.y,
                    self.config.sensor_height_m,
                    pose.yaw + azimuth,
                    elevation,
                    self.config.max_range_m,
                    timestamp,
                )
                for azimuth, elevation in zip(self.azimuths, self.elevations, strict=True)
            ],
            dtype=float,
        )
        return LidarScan3D(timestamp, self.azimuths.copy(), self.elevations.copy(), ranges)

    def point_cloud(self, scan: LidarScan3D, pose: Pose2D) -> np.ndarray:
        """Return global XYZ points for finite, in-range observations only."""

        valid = np.isfinite(scan.ranges) & (scan.ranges < self.config.max_range_m - 1e-12)
        ranges = scan.ranges[valid]
        azimuths = scan.azimuths[valid] + pose.yaw
        elevations = scan.elevations[valid]
        horizontal = ranges * np.cos(elevations)
        return np.column_stack(
            (
                pose.x + horizontal * np.cos(azimuths),
                pose.y + horizontal * np.sin(azimuths),
                self.config.sensor_height_m + ranges * np.sin(elevations),
            )
        )
