"""Exact constant-twist integration for a differential-drive robot."""

from __future__ import annotations

import math

from slamcart3d.core.models import Pose2D, wrap_angle


class DifferentialDrive:
    """Kinematic differential-drive model with wheel angular-rate inputs."""

    def __init__(self, wheel_radius_m: float, wheel_base_m: float, pose: Pose2D) -> None:
        if wheel_radius_m <= 0.0 or wheel_base_m <= 0.0:
            raise ValueError("wheel dimensions must be positive")
        self.wheel_radius_m = wheel_radius_m
        self.wheel_base_m = wheel_base_m
        self.pose = pose

    def step(self, left_rate_rad_s: float, right_rate_rad_s: float, dt_s: float) -> Pose2D:
        """Advance using the closed-form unicycle solution over one time step."""

        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        velocity = self.wheel_radius_m * (right_rate_rad_s + left_rate_rad_s) / 2.0
        yaw_rate = self.wheel_radius_m * (right_rate_rad_s - left_rate_rad_s) / self.wheel_base_m
        x, y, yaw = self.pose.x, self.pose.y, self.pose.yaw
        if abs(yaw_rate) < 1e-12:
            x += velocity * dt_s * math.cos(yaw)
            y += velocity * dt_s * math.sin(yaw)
        else:
            new_yaw = yaw + yaw_rate * dt_s
            radius = velocity / yaw_rate
            x += radius * (math.sin(new_yaw) - math.sin(yaw))
            y -= radius * (math.cos(new_yaw) - math.cos(yaw))
        self.pose = Pose2D(x, y, float(wrap_angle(yaw + yaw_rate * dt_s)))
        return self.pose

    def wheel_rates(self, linear_velocity_m_s: float, yaw_rate_rad_s: float) -> tuple[float, float]:
        """Convert a desired body twist into left/right wheel angular rates."""

        left = (linear_velocity_m_s - 0.5 * self.wheel_base_m * yaw_rate_rad_s) / self.wheel_radius_m
        right = (linear_velocity_m_s + 0.5 * self.wheel_base_m * yaw_rate_rad_s) / self.wheel_radius_m
        return left, right
