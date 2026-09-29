"""Reproducible degradation engine kept independent from ideal simulation."""

from __future__ import annotations

import numpy as np

from slamcart3d.core.config import DegradationConfig, IMUConfig
from slamcart3d.core.models import CameraFrame, IMUSample, LidarScan, LidarScan3D, Pose2D


class DegradationEngine:
    """Apply configured noise, bias, dropout, and outliers with one seeded RNG."""

    def __init__(self, config: DegradationConfig, seed: int) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        # Keep the LiDAR/odometry sequence unchanged when IMU is enabled.
        self._imu_rng = np.random.default_rng(seed + 2_000_003)
        self._imu_aux_rng = np.random.default_rng(seed + 2_000_019)
        self._lidar3d_rng = np.random.default_rng(seed + 3_000_007)
        self._camera_rng = np.random.default_rng(seed + 4_000_009)
        self._gyro_bias = config.imu_gyro_bias_rad_s
        self._accel_bias = config.imu_accel_bias_m_s2
        self._gyro_aux_bias = np.zeros(2, dtype=float)
        self._accel_aux_bias = np.zeros(2, dtype=float)

    def _dropout_probability(
        self, configured_probability: float, progress: float, pose: Pose2D | None
    ) -> float:
        """Combine time profile and a ground-truth-frame rectangular region."""

        if not 0.0 <= progress <= 1.0:
            raise ValueError("experiment progress must be between 0 and 1")
        if self.config.dropout_region_enabled:
            if pose is None:
                raise ValueError("spatial dropout requires a robot pose")
            if not (
                self.config.dropout_region_x_min_m <= pose.x <= self.config.dropout_region_x_max_m
                and self.config.dropout_region_y_min_m <= pose.y <= self.config.dropout_region_y_max_m
            ):
                return 0.0
        profile = self.config.dropout_time_profile
        if profile == "linear_ramp":
            return configured_probability * progress
        if profile == "abrupt":
            return configured_probability * (progress >= self.config.dropout_transition_fraction)
        return configured_probability

    def _degrade_ranges(
        self,
        ideal_ranges: np.ndarray,
        azimuths: np.ndarray,
        max_range_m: float,
        progress: float,
        rng: np.random.Generator,
        pose: Pose2D | None,
    ) -> np.ndarray:
        """Apply the shared LiDAR model with a stream-specific random generator."""

        ranges = ideal_ranges.copy()
        ranges += self.config.lidar_bias_m
        ranges += rng.normal(0.0, self.config.lidar_noise_std_m, size=ranges.shape)
        dropout = rng.random(ranges.shape) < self._dropout_probability(
            self.config.lidar_dropout_probability, progress, pose
        )
        dropout |= ideal_ranges >= max_range_m - 1e-12
        outliers = rng.random(ranges.shape) < self.config.lidar_outlier_probability
        ranges[outliers] = rng.uniform(0.05, max_range_m, size=int(np.count_nonzero(outliers)))
        effective_range = max_range_m * (1.0 - self.config.lidar_range_reduction_fraction)
        dropout |= ideal_ranges > effective_range
        if self.config.lidar_range_reduction_fraction > 0.0:
            dropout |= ranges > effective_range
        sector_width = self.config.lidar_sector_dropout_width_deg
        if sector_width > 0.0:
            center = np.deg2rad(self.config.lidar_sector_dropout_center_deg)
            wrapped_difference = (azimuths - center + np.pi) % (2.0 * np.pi) - np.pi
            dropout |= np.abs(wrapped_difference) <= np.deg2rad(sector_width) / 2.0 + 1e-12
        ranges[dropout] = np.nan
        ranges = np.where(np.isnan(ranges), ranges, np.clip(ranges, 0.0, max_range_m))
        return ranges

    def degrade_lidar(
        self, scan: LidarScan, max_range_m: float, progress: float = 1.0, pose: Pose2D | None = None
    ) -> LidarScan:
        """Return a degraded copy without modifying ground-truth ranges."""

        ranges = self._degrade_ranges(scan.ranges, scan.angles, max_range_m, progress, self.rng, pose)
        return LidarScan(timestamp=scan.timestamp, angles=scan.angles.copy(), ranges=ranges)

    def degrade_lidar3d(
        self, scan: LidarScan3D, max_range_m: float, progress: float = 1.0, pose: Pose2D | None = None
    ) -> LidarScan3D:
        """Degrade multilayer ranges without perturbing the planar sensor stream."""

        ranges = self._degrade_ranges(
            scan.ranges, scan.azimuths, max_range_m, progress, self._lidar3d_rng, pose
        )
        return LidarScan3D(scan.timestamp, scan.azimuths.copy(), scan.elevations.copy(), ranges)

    def degrade_wheel_rates(
        self, left: float, right: float, progress: float = 1.0, pose: Pose2D | None = None
    ) -> tuple[float, float]:
        """Simulate encoder bias/noise and complete odometry packet dropout."""

        if self.rng.random() < self._dropout_probability(
            self.config.odometry_dropout_probability, progress, pose
        ):
            return 0.0, 0.0
        noise = self.rng.normal(0.0, self.config.wheel_rate_noise_std_rad_s, size=2)
        bias = self.config.wheel_rate_bias_rad_s
        return float(left + bias + noise[0]), float(right + bias + noise[1])

    def degrade_imu(
        self,
        sample: IMUSample,
        timestep_s: float,
        imu: IMUConfig,
        progress: float = 1.0,
        pose: Pose2D | None = None,
    ) -> IMUSample:
        """Add bias random walk, white noise, saturation and packet dropout."""

        if timestep_s <= 0.0:
            raise ValueError("timestep_s must be positive")
        root_dt = float(np.sqrt(timestep_s))
        self._gyro_bias += float(
            self._imu_rng.normal(0.0, self.config.imu_gyro_random_walk_std_rad_s_sqrt_s * root_dt)
        )
        self._accel_bias += float(
            self._imu_rng.normal(0.0, self.config.imu_accel_random_walk_std_m_s2_sqrt_s * root_dt)
        )
        yaw_rate = (
            sample.yaw_rate_rad_s
            + self._gyro_bias
            + float(self._imu_rng.normal(0.0, self.config.imu_gyro_noise_std_rad_s))
        )
        acceleration = (
            sample.longitudinal_acceleration_m_s2
            + self._accel_bias
            + float(self._imu_rng.normal(0.0, self.config.imu_accel_noise_std_m_s2))
        )
        self._gyro_aux_bias += self._imu_aux_rng.normal(
            0.0, self.config.imu_gyro_random_walk_std_rad_s_sqrt_s * root_dt, size=2
        )
        self._accel_aux_bias += self._imu_aux_rng.normal(
            0.0, self.config.imu_accel_random_walk_std_m_s2_sqrt_s * root_dt, size=2
        )
        auxiliary_gyro = (
            np.array([sample.roll_rate_rad_s, sample.pitch_rate_rad_s])
            + self._gyro_aux_bias
            + self._imu_aux_rng.normal(0.0, self.config.imu_gyro_noise_std_rad_s, size=2)
        )
        auxiliary_acceleration = (
            np.array([sample.lateral_acceleration_m_s2, sample.vertical_specific_force_m_s2])
            + self._accel_aux_bias
            + self._imu_aux_rng.normal(0.0, self.config.imu_accel_noise_std_m_s2, size=2)
        )
        missing = self._imu_rng.random() < self._dropout_probability(
            self.config.imu_dropout_probability, progress, pose
        )
        return IMUSample(
            timestamp=sample.timestamp,
            yaw_rate_rad_s=float("nan")
            if missing
            else float(np.clip(yaw_rate, -imu.gyro_saturation_rad_s, imu.gyro_saturation_rad_s)),
            longitudinal_acceleration_m_s2=float("nan")
            if missing
            else float(np.clip(acceleration, -imu.accel_saturation_m_s2, imu.accel_saturation_m_s2)),
            roll_rate_rad_s=float("nan")
            if missing
            else float(np.clip(auxiliary_gyro[0], -imu.gyro_saturation_rad_s, imu.gyro_saturation_rad_s)),
            pitch_rate_rad_s=float("nan")
            if missing
            else float(np.clip(auxiliary_gyro[1], -imu.gyro_saturation_rad_s, imu.gyro_saturation_rad_s)),
            lateral_acceleration_m_s2=float("nan")
            if missing
            else float(
                np.clip(auxiliary_acceleration[0], -imu.accel_saturation_m_s2, imu.accel_saturation_m_s2)
            ),
            vertical_specific_force_m_s2=float("nan")
            if missing
            else float(
                np.clip(auxiliary_acceleration[1], -imu.accel_saturation_m_s2, imu.accel_saturation_m_s2)
            ),
        )

    def degrade_camera(
        self, frame: CameraFrame, progress: float = 1.0, pose: Pose2D | None = None
    ) -> CameraFrame | None:
        """Apply RGB/depth degradation using a camera-only random stream."""

        if self._camera_rng.random() < self._dropout_probability(
            self.config.camera_frame_dropout_probability, progress, pose
        ):
            return None
        rgb = frame.rgb.astype(float)
        mean_color = np.mean(rgb, axis=(0, 1), keepdims=True)
        rgb = mean_color + (rgb - mean_color) * (1.0 - self.config.camera_texture_reduction_fraction)
        rgb *= self.config.camera_lighting_factor
        radius = self.config.camera_blur_radius_px
        if radius:
            padded = np.pad(rgb, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
            blurred = np.zeros_like(rgb)
            for dy in range(2 * radius + 1):
                for dx in range(2 * radius + 1):
                    blurred += padded[dy : dy + len(rgb), dx : dx + rgb.shape[1]]
            rgb = blurred / (2 * radius + 1) ** 2
        rgb += self._camera_rng.normal(0.0, self.config.camera_rgb_noise_std, size=rgb.shape)
        depth = frame.depth_m.copy()
        valid = np.isfinite(depth)
        depth[valid] = np.clip(
            depth[valid]
            + self._camera_rng.normal(0.0, self.config.camera_depth_noise_std_m, size=int(np.sum(valid))),
            0.0,
            None,
        )
        occlusion = self.config.camera_occlusion_fraction
        if occlusion > 0.0:
            height, width = depth.shape
            box_width = min(width, max(1, int(np.ceil(np.sqrt(occlusion) * width))))
            box_height = min(height, max(1, int(np.ceil(occlusion * width * height / box_width))))
            left = int(self._camera_rng.integers(0, width - box_width + 1))
            top = int(self._camera_rng.integers(0, height - box_height + 1))
            rgb[top : top + box_height, left : left + box_width] = 0.0
            depth[top : top + box_height, left : left + box_width] = np.nan
        return CameraFrame(frame.timestamp, np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8), depth)
