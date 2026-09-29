"""Six-axis IMU observations constrained to a planar kinematic robot."""

from __future__ import annotations

from slamcart3d.core.models import IMUSample


class PlanarIMU:
    """Expose 3-axis angular rate and specific force for level-ground motion."""

    STANDARD_GRAVITY_M_S2 = 9.80665

    def sample(
        self,
        timestamp_s: float,
        linear_velocity_m_s: float,
        previous_linear_velocity_m_s: float,
        yaw_rate_rad_s: float,
        timestep_s: float,
    ) -> IMUSample:
        """Sample a piecewise-constant commanded twist at one time step."""

        if timestep_s <= 0.0:
            raise ValueError("timestep_s must be positive")
        return IMUSample(
            timestamp=timestamp_s,
            yaw_rate_rad_s=yaw_rate_rad_s,
            longitudinal_acceleration_m_s2=(linear_velocity_m_s - previous_linear_velocity_m_s) / timestep_s,
            roll_rate_rad_s=0.0,
            pitch_rate_rad_s=0.0,
            lateral_acceleration_m_s2=linear_velocity_m_s * yaw_rate_rad_s,
            vertical_specific_force_m_s2=self.STANDARD_GRAVITY_M_S2,
        )
