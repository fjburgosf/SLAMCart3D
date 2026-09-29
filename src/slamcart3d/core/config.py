"""Validated experiment configuration with YAML/JSON serialization."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RobotConfig:
    """Differential-drive geometry in SI units."""

    wheel_radius_m: float = 0.1
    wheel_base_m: float = 0.5
    initial_x_m: float = 5.0
    initial_y_m: float = 4.0
    initial_yaw_rad: float = 0.0


@dataclass(frozen=True)
class LidarConfig:
    """Planar LiDAR sampling parameters."""

    beam_count: int = 90
    field_of_view_deg: float = 270.0
    max_range_m: float = 12.0


@dataclass(frozen=True)
class Lidar3DConfig:
    """Optional multielevation LiDAR for synthetic 3D point observations."""

    enabled: bool = False
    azimuth_beams: int = 36
    elevation_beams: int = 4
    horizontal_fov_deg: float = 270.0
    vertical_fov_deg: float = 30.0
    sensor_height_m: float = 0.5
    max_range_m: float = 12.0


@dataclass(frozen=True)
class CameraConfig:
    """Optional low-resolution perspective RGB-D camera for synthetic data."""

    enabled: bool = False
    width_px: int = 64
    height_px: int = 48
    horizontal_fov_deg: float = 90.0
    vertical_fov_deg: float = 60.0
    sensor_height_m: float = 0.5
    max_range_m: float = 12.0


@dataclass(frozen=True)
class IMUConfig:
    """Planar IMU availability and symmetric measurement limits."""

    enabled: bool = True
    gyro_saturation_rad_s: float = 6.0
    accel_saturation_m_s2: float = 20.0


@dataclass(frozen=True)
class DegradationConfig:
    """Independent, reproducible sensor and odometry degradations."""

    lidar_noise_std_m: float = 0.02
    lidar_bias_m: float = 0.0
    lidar_dropout_probability: float = 0.0
    lidar_outlier_probability: float = 0.0
    lidar_range_reduction_fraction: float = 0.0
    lidar_sector_dropout_center_deg: float = 0.0
    lidar_sector_dropout_width_deg: float = 0.0
    wheel_rate_noise_std_rad_s: float = 0.01
    wheel_rate_bias_rad_s: float = 0.0
    odometry_dropout_probability: float = 0.0
    imu_gyro_bias_rad_s: float = 0.0
    imu_gyro_noise_std_rad_s: float = 0.0
    imu_gyro_random_walk_std_rad_s_sqrt_s: float = 0.0
    imu_accel_bias_m_s2: float = 0.0
    imu_accel_noise_std_m_s2: float = 0.0
    imu_accel_random_walk_std_m_s2_sqrt_s: float = 0.0
    imu_dropout_probability: float = 0.0
    camera_rgb_noise_std: float = 0.0
    camera_depth_noise_std_m: float = 0.0
    camera_blur_radius_px: int = 0
    camera_lighting_factor: float = 1.0
    camera_texture_reduction_fraction: float = 0.0
    camera_occlusion_fraction: float = 0.0
    camera_frame_dropout_probability: float = 0.0
    dropout_time_profile: str = "constant"
    dropout_transition_fraction: float = 0.5
    dropout_region_enabled: bool = False
    dropout_region_x_min_m: float = 0.0
    dropout_region_x_max_m: float = 1.0
    dropout_region_y_min_m: float = 0.0
    dropout_region_y_max_m: float = 1.0


@dataclass(frozen=True)
class MotionSegment:
    """One constant-twist interval in a user-defined motion sequence."""

    duration_s: float
    linear_velocity_m_s: float
    yaw_rate_rad_s: float


@dataclass(frozen=True)
class SimulationConfig:
    """Time integration and commanded motion settings."""

    duration_s: float = 20.0
    timestep_s: float = 0.1
    linear_velocity_m_s: float = 0.4
    yaw_rate_rad_s: float = 0.2
    motion_profile: str = "constant"
    motion_frequency_hz: float = 0.1
    motion_segments: tuple[MotionSegment, ...] = ()
    world: str = "indoor_simple"


@dataclass(frozen=True)
class SLAMConfig:
    """Algorithm selection and algorithm-specific safe defaults."""

    algorithm: str = "dead_reckoning"
    icp_max_iterations: int = 15
    icp_max_correspondence_m: float = 0.6
    icp_prior_weight: float = 20.0


@dataclass(frozen=True)
class RobustnessScoreConfig:
    """Optional, explicitly weighted and normalized robustness score."""

    enabled: bool = False
    ate_half_quality_m: float = 0.1
    rpe_translation_half_quality_m: float = 0.05
    map_f1_floor: float = 0.5
    tracking_availability_floor: float = 0.9
    ate_weight: float = 1.0
    rpe_translation_weight: float = 1.0
    map_f1_weight: float = 1.0
    tracking_availability_weight: float = 1.0


@dataclass(frozen=True)
class OccupancyGridConfig:
    """Optional observed free/occupied mapping from planar LiDAR rays."""

    enabled: bool = False
    resolution_m: float = 0.2
    beam_stride: int = 5


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete input required to reproduce one experiment."""

    name: str = "example_01_clean_environment"
    seed: int = 42
    robot: RobotConfig = field(default_factory=RobotConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    lidar3d: Lidar3DConfig = field(default_factory=Lidar3DConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    slam: SLAMConfig = field(default_factory=SLAMConfig)
    robustness_score: RobustnessScoreConfig = field(default_factory=RobustnessScoreConfig)
    occupancy_grid: OccupancyGridConfig = field(default_factory=OccupancyGridConfig)

    def __post_init__(self) -> None:
        if any(not isinstance(segment, MotionSegment) for segment in self.simulation.motion_segments):
            raise ValueError("motion_segments must contain MotionSegment values")
        numeric_values = (
            self.robot.wheel_radius_m,
            self.robot.wheel_base_m,
            self.robot.initial_x_m,
            self.robot.initial_y_m,
            self.robot.initial_yaw_rad,
            self.lidar.field_of_view_deg,
            self.lidar.max_range_m,
            self.lidar3d.horizontal_fov_deg,
            self.lidar3d.vertical_fov_deg,
            self.lidar3d.sensor_height_m,
            self.lidar3d.max_range_m,
            self.camera.horizontal_fov_deg,
            self.camera.vertical_fov_deg,
            self.camera.sensor_height_m,
            self.camera.max_range_m,
            self.imu.gyro_saturation_rad_s,
            self.imu.accel_saturation_m_s2,
            *(
                value
                for name, value in self.degradation.__dict__.items()
                if name not in {"dropout_time_profile", "dropout_region_enabled"}
            ),
            self.simulation.duration_s,
            self.simulation.timestep_s,
            self.simulation.linear_velocity_m_s,
            self.simulation.yaw_rate_rad_s,
            self.simulation.motion_frequency_hz,
            *(
                value
                for segment in self.simulation.motion_segments
                for value in (segment.duration_s, segment.linear_velocity_m_s, segment.yaw_rate_rad_s)
            ),
            self.slam.icp_max_correspondence_m,
            self.slam.icp_prior_weight,
            self.occupancy_grid.resolution_m,
            *(value for name, value in self.robustness_score.__dict__.items() if name != "enabled"),
        )
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("configuration values must be finite")
        if self.simulation.timestep_s <= 0.0 or self.simulation.duration_s <= 0.0:
            raise ValueError("duration_s and timestep_s must be positive")
        if self.simulation.motion_profile not in {"constant", "sinusoidal_yaw", "stop_turn", "scripted"}:
            raise ValueError(f"unsupported motion profile: {self.simulation.motion_profile}")
        if self.simulation.motion_frequency_hz <= 0.0:
            raise ValueError("motion_frequency_hz must be positive")
        steps = self.simulation.duration_s / self.simulation.timestep_s
        if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("duration_s must be an integer multiple of timestep_s")
        segment_steps = [
            segment.duration_s / self.simulation.timestep_s for segment in self.simulation.motion_segments
        ]
        if any(
            segment.duration_s <= 0.0 or not math.isclose(count, round(count), rel_tol=0.0, abs_tol=1e-9)
            for segment, count in zip(self.simulation.motion_segments, segment_steps, strict=True)
        ):
            raise ValueError("each motion segment duration must be positive and a multiple of timestep_s")
        if self.simulation.motion_profile == "scripted" and (
            not segment_steps or sum(round(count) for count in segment_steps) != round(steps)
        ):
            raise ValueError("scripted motion segments must exactly cover duration_s")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.robot.wheel_radius_m <= 0.0 or self.robot.wheel_base_m <= 0.0:
            raise ValueError("wheel geometry must be positive")
        if self.lidar.beam_count < 2 or self.lidar.max_range_m <= 0.0:
            raise ValueError("LiDAR requires at least two beams and a positive range")
        if not 0.0 < self.lidar.field_of_view_deg <= 360.0:
            raise ValueError("LiDAR field of view must be in (0, 360] degrees")
        if self.lidar3d.azimuth_beams < 2 or self.lidar3d.elevation_beams < 2:
            raise ValueError("3D LiDAR requires at least two azimuth and elevation beams")
        if not 0.0 < self.lidar3d.horizontal_fov_deg <= 360.0:
            raise ValueError("3D LiDAR horizontal field of view must be in (0, 360]")
        if not 0.0 < self.lidar3d.vertical_fov_deg < 180.0:
            raise ValueError("3D LiDAR vertical field of view must be in (0, 180)")
        if self.lidar3d.sensor_height_m <= 0.0 or self.lidar3d.max_range_m <= 0.0:
            raise ValueError("3D LiDAR height and range must be positive")
        if self.camera.width_px < 2 or self.camera.height_px < 2:
            raise ValueError("camera image width and height must be at least two pixels")
        if (
            not isinstance(self.camera.width_px, int)
            or not isinstance(self.camera.height_px, int)
            or self.camera.width_px > 160
            or self.camera.height_px > 120
        ):
            raise ValueError("camera resolution must be integer and at most 160 x 120 pixels")
        if not 0.0 < self.camera.horizontal_fov_deg < 180.0:
            raise ValueError("camera horizontal field of view must be in (0, 180)")
        if not 0.0 < self.camera.vertical_fov_deg < 180.0:
            raise ValueError("camera vertical field of view must be in (0, 180)")
        if self.camera.sensor_height_m <= 0.0 or self.camera.max_range_m <= 0.0:
            raise ValueError("camera height and range must be positive")
        if self.imu.gyro_saturation_rad_s <= 0.0 or self.imu.accel_saturation_m_s2 <= 0.0:
            raise ValueError("IMU saturation limits must be positive")
        if self.slam.algorithm not in {"dead_reckoning", "icp_odometry", "wheel_imu_fusion"}:
            raise ValueError(f"unsupported SLAM algorithm: {self.slam.algorithm}")
        if self.slam.algorithm == "wheel_imu_fusion" and not self.imu.enabled:
            raise ValueError("wheel_imu_fusion requires an enabled IMU")
        if not 0.05 <= self.occupancy_grid.resolution_m <= 1.0:
            raise ValueError("occupancy grid resolution must be between 0.05 and 1.0 m")
        if not isinstance(self.occupancy_grid.beam_stride, int) or self.occupancy_grid.beam_stride < 1:
            raise ValueError("occupancy grid beam stride must be a positive integer")
        if (
            self.slam.icp_max_iterations < 1
            or self.slam.icp_max_correspondence_m <= 0.0
            or self.slam.icp_prior_weight <= 0.0
        ):
            raise ValueError("ICP iterations, correspondence radius, and prior weight must be positive")
        if any(
            value < 0.0
            for value in (
                self.degradation.lidar_noise_std_m,
                self.degradation.wheel_rate_noise_std_rad_s,
                self.degradation.imu_gyro_noise_std_rad_s,
                self.degradation.imu_gyro_random_walk_std_rad_s_sqrt_s,
                self.degradation.imu_accel_noise_std_m_s2,
                self.degradation.imu_accel_random_walk_std_m_s2_sqrt_s,
                self.degradation.camera_rgb_noise_std,
                self.degradation.camera_depth_noise_std_m,
            )
        ):
            raise ValueError("noise standard deviations must be non-negative")
        probabilities = (
            self.degradation.lidar_dropout_probability,
            self.degradation.lidar_outlier_probability,
            self.degradation.lidar_range_reduction_fraction,
            self.degradation.odometry_dropout_probability,
            self.degradation.imu_dropout_probability,
            self.degradation.camera_texture_reduction_fraction,
            self.degradation.camera_occlusion_fraction,
            self.degradation.camera_frame_dropout_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("probabilities must be between 0 and 1")
        if self.degradation.dropout_time_profile not in {"constant", "linear_ramp", "abrupt"}:
            raise ValueError("dropout_time_profile must be constant, linear_ramp, or abrupt")
        if (
            not isinstance(self.degradation.camera_blur_radius_px, int)
            or not 0 <= self.degradation.camera_blur_radius_px <= 5
        ):
            raise ValueError("camera_blur_radius_px must be between 0 and 5")
        if self.degradation.camera_lighting_factor <= 0.0 or self.degradation.camera_lighting_factor > 2.0:
            raise ValueError("camera_lighting_factor must be in (0, 2]")
        if not 0.0 <= self.degradation.dropout_transition_fraction <= 1.0:
            raise ValueError("dropout_transition_fraction must be between 0 and 1")
        if self.degradation.dropout_region_enabled and (
            self.degradation.dropout_region_x_min_m >= self.degradation.dropout_region_x_max_m
            or self.degradation.dropout_region_y_min_m >= self.degradation.dropout_region_y_max_m
        ):
            raise ValueError("enabled dropout region requires positive x and y spans")
        if not 0.0 <= self.degradation.lidar_sector_dropout_width_deg <= 360.0:
            raise ValueError("LiDAR sector dropout width must be between 0 and 360 degrees")
        score = self.robustness_score
        if score.ate_half_quality_m <= 0.0 or score.rpe_translation_half_quality_m <= 0.0:
            raise ValueError("SRS half-quality error scales must be positive")
        if not 0.0 <= score.map_f1_floor < 1.0 or not 0.0 <= score.tracking_availability_floor < 1.0:
            raise ValueError("SRS quality floors must be in [0, 1)")
        weights = (
            score.ate_weight,
            score.rpe_translation_weight,
            score.map_f1_weight,
            score.tracking_availability_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("SRS weights must be nonnegative and have a positive sum")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-safe configuration dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Build a configuration while rejecting no known nested sections."""

        simulation_data = dict(data.get("simulation", {}))
        if "motion_segments" in simulation_data:
            segments = simulation_data["motion_segments"]
            if not isinstance(segments, (list, tuple)) or any(
                not isinstance(item, dict) for item in segments
            ):
                raise ValueError("motion_segments must be a sequence of mappings")
            simulation_data["motion_segments"] = tuple(MotionSegment(**item) for item in segments)
        return cls(
            name=data.get("name", cls.name),
            seed=int(data.get("seed", cls.seed)),
            robot=RobotConfig(**data.get("robot", {})),
            lidar=LidarConfig(**data.get("lidar", {})),
            lidar3d=Lidar3DConfig(**data.get("lidar3d", {})),
            camera=CameraConfig(**data.get("camera", {})),
            imu=IMUConfig(**data.get("imu", {})),
            degradation=DegradationConfig(**data.get("degradation", {})),
            simulation=SimulationConfig(**simulation_data),
            slam=SLAMConfig(**data.get("slam", {})),
            robustness_score=RobustnessScoreConfig(**data.get("robustness_score", {})),
            occupancy_grid=OccupancyGridConfig(**data.get("occupancy_grid", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """Load YAML or JSON from *path*."""

        source = Path(path)
        with source.open("r", encoding="utf-8") as stream:
            data = json.load(stream) if source.suffix.lower() == ".json" else yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise ValueError("experiment configuration must be a mapping")
        return cls.from_dict(data.get("experiment", data))
