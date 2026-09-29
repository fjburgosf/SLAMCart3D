"""Small immutable data structures shared by the simulation modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angles to the half-open interval [-pi, pi)."""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class Pose2D:
    """Planar pose embedded in a 3D world at ground height."""

    x: float
    y: float
    yaw: float

    def as_array(self) -> np.ndarray:
        """Return ``[x, y, yaw]`` as a floating-point array."""

        return np.array([self.x, self.y, self.yaw], dtype=float)


@dataclass(frozen=True)
class LidarScan:
    """One planar LiDAR scan with angles expressed in the robot frame."""

    timestamp: float
    angles: np.ndarray
    ranges: np.ndarray


@dataclass(frozen=True)
class LidarScan3D:
    """Flattened multilayer rays with robot-frame azimuth/elevation and slant range."""

    timestamp: float
    azimuths: np.ndarray
    elevations: np.ndarray
    ranges: np.ndarray


@dataclass(frozen=True)
class CameraFrame:
    """Perspective RGB image and metric slant-depth array; NaN is no hit."""

    timestamp: float
    rgb: np.ndarray
    depth_m: np.ndarray


@dataclass(frozen=True)
class IMUSample:
    """Body-frame six-axis IMU sample on a planar robot; NaN means missing."""

    timestamp: float
    yaw_rate_rad_s: float
    longitudinal_acceleration_m_s2: float
    roll_rate_rad_s: float = 0.0
    pitch_rate_rad_s: float = 0.0
    lateral_acceleration_m_s2: float = 0.0
    vertical_specific_force_m_s2: float = 0.0
