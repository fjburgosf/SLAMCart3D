"""Reliability-adaptive wheel/IMU and scan-to-map localization with online mapping."""

from __future__ import annotations

import math

import numpy as np

from slamcart3d.core.models import IMUSample, LidarScan, Pose2D, wrap_angle
from slamcart3d.robots.differential_drive import DifferentialDrive
from slamcart3d.slam.base import SLAMAlgorithm


def _register_scan_to_map(
    scan_points: np.ndarray,
    map_points: np.ndarray,
    prior: Pose2D,
    prior_weight: float,
    max_correspondence_m: float,
    max_iterations: int,
) -> tuple[Pose2D | None, float, float]:
    """Robust planar point-to-point alignment with a wheel/IMU pose prior."""

    state = np.array([prior.x, prior.y, prior.yaw], dtype=float)
    reference = state.copy()
    if len(scan_points) < 6 or len(map_points) < 6:
        return None, math.inf, 0.0

    for _ in range(max_iterations):
        cosine, sine = math.cos(state[2]), math.sin(state[2])
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        transformed = scan_points @ rotation.T + state[:2]
        distances_sq = np.sum((transformed[:, None, :] - map_points[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distances_sq, axis=1)
        distances = np.sqrt(distances_sq[np.arange(len(scan_points)), nearest])
        valid = distances <= max_correspondence_m
        if np.count_nonzero(valid) < 6:
            return None, math.inf, float(np.mean(valid))
        selected = scan_points[valid]
        residual = transformed[valid] - map_points[nearest[valid]]
        robust_weight = np.minimum(1.0, 0.15 / np.maximum(distances[valid], 1e-9))
        jacobian = np.zeros((len(selected), 2, 3), dtype=float)
        jacobian[:, 0, 0] = 1.0
        jacobian[:, 1, 1] = 1.0
        jacobian[:, 0, 2] = -sine * selected[:, 0] - cosine * selected[:, 1]
        jacobian[:, 1, 2] = cosine * selected[:, 0] - sine * selected[:, 1]
        hessian = np.einsum("n,nij,nik->jk", robust_weight, jacobian, jacobian)
        gradient = np.einsum("n,nij,ni->j", robust_weight, jacobian, residual)
        deviation = state - reference
        deviation[2] = float(wrap_angle(deviation[2]))
        hessian += prior_weight * np.eye(3)
        gradient += prior_weight * deviation
        step = -np.linalg.solve(hessian, gradient)
        step[:2] = np.clip(step[:2], -0.15, 0.15)
        step[2] = float(np.clip(step[2], -0.08, 0.08))
        state += step
        state[2] = float(wrap_angle(state[2]))
        if float(np.linalg.norm(step)) < 1e-5:
            break

    cosine, sine = math.cos(state[2]), math.sin(state[2])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    transformed = scan_points @ rotation.T + state[:2]
    distances_sq = np.sum((transformed[:, None, :] - map_points[None, :, :]) ** 2, axis=2)
    distances = np.sqrt(np.min(distances_sq, axis=1))
    valid = distances <= max_correspondence_m
    if np.count_nonzero(valid) < 6:
        return None, math.inf, float(np.mean(valid))
    rmse = float(np.sqrt(np.mean(np.square(distances[valid]))))
    return Pose2D(float(state[0]), float(state[1]), float(state[2])), rmse, float(np.mean(valid))


class AdaptiveHybridSLAM(SLAMAlgorithm):
    """Project method: scan-to-map correction with online reliability learning.

    The EMA uses only measured correspondence quality; it never sees true pose.
    No global pose-graph optimization or loop closure is claimed.
    """

    def __init__(
        self,
        wheel_radius_m: float,
        wheel_base_m: float,
        lidar_max_range_m: float,
        max_iterations: int = 8,
        max_correspondence_m: float = 0.6,
        prior_weight: float = 20.0,
        use_imu: bool = True,
        adaptive_trust: bool = True,
        gate_map: bool = True,
    ) -> None:
        self.wheel_radius_m = wheel_radius_m
        self.wheel_base_m = wheel_base_m
        self.lidar_max_range_m = lidar_max_range_m
        self.max_iterations = max_iterations
        self.max_correspondence_m = max_correspondence_m
        self.prior_weight = prior_weight
        self.use_imu = use_imu
        self.adaptive_trust = adaptive_trust
        self.gate_map = gate_map
        self._pose: Pose2D | None = None
        self._chunks: list[np.ndarray] = []
        self._updates = 0
        self._fallbacks = 0
        self._accepted = 0
        self._imu_updates = 0
        self._trust = 0.5
        self._last_scan_rmse: float | None = None

    def initialize(self, pose: Pose2D) -> None:
        self._pose = pose
        self._chunks = []
        self._updates = 0
        self._fallbacks = 0
        self._accepted = 0
        self._imu_updates = 0
        self._trust = 0.5
        self._last_scan_rmse = None

    def _scan_points(self, scan: LidarScan) -> np.ndarray:
        valid = (
            np.isfinite(scan.ranges) & (scan.ranges > 0.02) & (scan.ranges < 0.995 * self.lidar_max_range_m)
        )
        selected = np.flatnonzero(valid)[::2]
        return np.column_stack(
            (
                scan.ranges[selected] * np.cos(scan.angles[selected]),
                scan.ranges[selected] * np.sin(scan.angles[selected]),
            )
        )

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
        if self._pose is None:
            raise RuntimeError("algorithm must be initialized before update")
        previous = self._pose
        velocity = self.wheel_radius_m * (left_rate + right_rate) / 2.0
        yaw_rate = self.wheel_radius_m * (right_rate - left_rate) / self.wheel_base_m
        if self.use_imu and imu is not None and np.isfinite(imu.yaw_rate_rad_s):
            yaw_rate = float(imu.yaw_rate_rad_s)
            self._imu_updates += 1
        predictor = DifferentialDrive(self.wheel_radius_m, self.wheel_base_m, previous)
        predicted = predictor.step(*predictor.wheel_rates(velocity, yaw_rate), dt_s)
        scan_points = self._scan_points(scan)
        accepted = False
        pose = predicted
        if self._chunks and len(scan_points) >= 6:
            map_points = self.get_map()
            nearby = (
                np.sum((map_points - [predicted.x, predicted.y]) ** 2, axis=1)
                <= (self.lidar_max_range_m * 1.2) ** 2
            )
            target = map_points[nearby]
            if len(target) > 250:
                target = target[:: int(np.ceil(len(target) / 250))]
            weight = self.prior_weight / max(self._trust, 0.2) if self.adaptive_trust else self.prior_weight
            candidate, rmse, fraction = _register_scan_to_map(
                scan_points,
                target,
                predicted,
                weight,
                self.max_correspondence_m,
                self.max_iterations,
            )
            quality = fraction * math.exp(-0.5 * (rmse / 0.25) ** 2) if math.isfinite(rmse) else 0.0
            if self.adaptive_trust:
                self._trust = 0.85 * self._trust + 0.15 * quality
            if candidate is not None and fraction >= 0.3 and rmse <= 0.3:
                correction_m = math.hypot(candidate.x - predicted.x, candidate.y - predicted.y)
                correction_yaw = abs(float(wrap_angle(candidate.yaw - predicted.yaw)))
                if correction_m <= 0.5 and correction_yaw <= 0.25:
                    pose = candidate
                    accepted = True
                    self._accepted += 1
                    self._last_scan_rmse = rmse
            if not accepted:
                self._fallbacks += 1
        elif self._chunks:
            self._fallbacks += 1
            if self.adaptive_trust:
                self._trust *= 0.85
        self._pose = pose
        if len(scan_points) and (not self._chunks or accepted or not self.gate_map):
            selected = scan_points[::2]
            cosine, sine = math.cos(pose.yaw), math.sin(pose.yaw)
            rotation = np.array([[cosine, -sine], [sine, cosine]])
            self._chunks.append(selected @ rotation.T + [pose.x, pose.y])
        self._updates += 1
        return pose

    def reset(self) -> None:
        self._pose = None
        self._chunks = []
        self._updates = 0
        self._fallbacks = 0
        self._accepted = 0
        self._imu_updates = 0
        self._trust = 0.5
        self._last_scan_rmse = None

    def get_pose(self) -> Pose2D:
        if self._pose is None:
            raise RuntimeError("algorithm must be initialized before update")
        return self._pose

    def get_map(self) -> np.ndarray:
        return np.vstack(self._chunks) if self._chunks else np.empty((0, 2), dtype=float)

    def get_diagnostics(self) -> dict[str, float | int | str | None]:
        return {
            "algorithm": "adaptive_hybrid",
            "updates": self._updates,
            "fallbacks": self._fallbacks,
            "accepted_map_registrations": self._accepted,
            "imu_updates": self._imu_updates,
            "online_trust": self._trust,
            "last_scan_rmse_m": self._last_scan_rmse,
            "map_points": len(self.get_map()),
        }
