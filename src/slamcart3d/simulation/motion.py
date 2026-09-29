"""Deterministic kinematic command profiles for motion-degradation studies."""

from __future__ import annotations

import math

from slamcart3d.core.config import SimulationConfig


def motion_command(config: SimulationConfig, step_index: int, total_steps: int) -> tuple[float, float]:
    """Return commanded forward speed and yaw rate for one integration step."""

    if not 0 <= step_index < total_steps:
        raise ValueError("step_index must be within the simulation")
    if config.motion_profile == "constant":
        return config.linear_velocity_m_s, config.yaw_rate_rad_s
    if config.motion_profile == "sinusoidal_yaw":
        time_s = step_index * config.timestep_s
        yaw_rate = config.yaw_rate_rad_s * math.cos(2.0 * math.pi * config.motion_frequency_hz * time_s)
        return config.linear_velocity_m_s, yaw_rate
    if config.motion_profile == "stop_turn":
        phase = 4 * step_index // total_steps
        if phase in (0, 2):
            return config.linear_velocity_m_s, 0.0
        return 0.0, config.yaw_rate_rad_s
    if config.motion_profile == "scripted":
        last_step = 0
        for segment in config.motion_segments:
            last_step += round(segment.duration_s / config.timestep_s)
            if step_index < last_step:
                return segment.linear_velocity_m_s, segment.yaw_rate_rad_s
        raise ValueError("scripted motion segments do not cover this step")
    raise ValueError(f"unsupported motion profile: {config.motion_profile}")
