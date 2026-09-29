"""Optional deterministic perspective RGB-D camera over the 2.5D world."""

from __future__ import annotations

import math

import numpy as np

from slamcart3d.core.config import CameraConfig
from slamcart3d.core.models import CameraFrame, Pose2D
from slamcart3d.worlds.geometry import World3D


class SyntheticCamera:
    """Ray-cast geometry with schematic colors, not photorealistic imaging."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        horizontal = math.tan(math.radians(config.horizontal_fov_deg) / 2.0)
        vertical = math.tan(math.radians(config.vertical_fov_deg) / 2.0)
        u = ((np.arange(config.width_px) + 0.5) * 2.0 / config.width_px - 1.0) * horizontal
        v = (1.0 - (np.arange(config.height_px) + 0.5) * 2.0 / config.height_px) * vertical
        uu, vv = np.meshgrid(u, v)
        self.azimuths = np.arctan2(uu, 1.0)
        self.elevations = np.arctan2(vv, np.sqrt(1.0 + uu * uu))

    def capture(self, timestamp: float, pose: Pose2D, world: World3D) -> CameraFrame:
        """Return ideal RGB and ray-depth arrays at a known robot pose."""

        rgb = np.empty((self.config.height_px, self.config.width_px, 3), dtype=np.uint8)
        depth = np.full((self.config.height_px, self.config.width_px), np.nan, dtype=float)
        for row in range(self.config.height_px):
            for column in range(self.config.width_px):
                azimuth = pose.yaw + float(self.azimuths[row, column])
                elevation = float(self.elevations[row, column])
                distance = world.ray_distance_3d(
                    pose.x,
                    pose.y,
                    self.config.sensor_height_m,
                    azimuth,
                    elevation,
                    self.config.max_range_m,
                    timestamp,
                )
                if distance >= self.config.max_range_m - 1e-12:
                    rgb[row, column] = (115, 178, 228)
                    continue
                depth[row, column] = distance
                horizontal = distance * math.cos(elevation)
                x = pose.x + horizontal * math.cos(azimuth)
                y = pose.y + horizontal * math.sin(azimuth)
                z = self.config.sensor_height_m + distance * math.sin(elevation)
                checker = (math.floor(2.0 * x) + math.floor(2.0 * y) + math.floor(2.0 * z)) % 2
                tone = 12 if checker else -12
                base = (114, 107, 93) if z <= 1e-7 else (145, 153, 161)
                rgb[row, column] = tuple(int(np.clip(value + tone, 0, 255)) for value in base)
        return CameraFrame(timestamp, rgb, depth)
