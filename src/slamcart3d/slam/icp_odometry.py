"""Scan-to-scan LiDAR odometry using point-to-point ICP."""

from __future__ import annotations

import math

import numpy as np

from slamcart3d.core.models import IMUSample, LidarScan, Pose2D, wrap_angle
from slamcart3d.robots.differential_drive import DifferentialDrive
from slamcart3d.slam.base import SLAMAlgorithm


def point_to_line_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    max_iterations: int = 15,
    max_correspondence_m: float = 0.6,
    prior_weight: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Register ordered scan points to local surfaces with an odometry prior.

    Adjacent target returns define local wall normals. The regularized
    point-to-line objective constrains translation along poorly observed
    surfaces, a common failure mode of point-to-point scan matching.
    """

    if len(source) < 3 or len(target) < 3:
        return initial_rotation, initial_translation, float("inf"), 0
    tangents = np.zeros_like(target)
    tangents[1:-1] = target[2:] - target[:-2]
    left_gaps = np.linalg.norm(target[1:-1] - target[:-2], axis=1)
    right_gaps = np.linalg.norm(target[2:] - target[1:-1], axis=1)
    usable = np.zeros(len(target), dtype=bool)
    usable[1:-1] = (left_gaps < 0.8) & (right_gaps < 0.8)
    norms = np.linalg.norm(tangents, axis=1)
    usable &= norms > 1e-8
    if np.count_nonzero(usable) < 3:
        return initial_rotation, initial_translation, float("inf"), 0
    normals = np.zeros_like(target)
    normals[usable, 0] = -tangents[usable, 1] / norms[usable]
    normals[usable, 1] = tangents[usable, 0] / norms[usable]
    initial = np.array(
        [
            initial_translation[0],
            initial_translation[1],
            math.atan2(initial_rotation[1, 0], initial_rotation[0, 0]),
        ]
    )
    parameters = initial.copy()
    inlier_count = 0
    error = float("inf")
    for _ in range(max_iterations):
        x_shift, y_shift, angle = parameters
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        transformed = source @ rotation.T + [x_shift, y_shift]
        squared = np.sum((transformed[:, None, :] - target[None, :, :]) ** 2, axis=2)
        indices = np.argmin(squared, axis=1)
        distances = np.sqrt(squared[np.arange(len(source)), indices])
        inliers = (distances <= max_correspondence_m) & usable[indices]
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 3:
            return initial_rotation, initial_translation, float("inf"), inlier_count
        points = source[inliers]
        selected_normals = normals[indices[inliers]]
        residuals = np.sum(selected_normals * (transformed[inliers] - target[indices[inliers]]), axis=1)
        yaw_derivative = np.column_stack(
            (-sine * points[:, 0] - cosine * points[:, 1], cosine * points[:, 0] - sine * points[:, 1])
        )
        jacobian = np.column_stack(
            (
                selected_normals[:, 0],
                selected_normals[:, 1],
                np.sum(selected_normals * yaw_derivative, axis=1),
            )
        )
        damping = np.diag([prior_weight, prior_weight, prior_weight])
        update = np.linalg.solve(
            jacobian.T @ jacobian + damping,
            -jacobian.T @ residuals - damping @ (parameters - initial),
        )
        parameters += update
        error = float(np.sqrt(np.mean(np.square(residuals))))
        if np.linalg.norm(update) < 1e-6:
            break
    cosine, sine = math.cos(parameters[2]), math.sin(parameters[2])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return rotation, parameters[:2].copy(), error, inlier_count


class ICPOdometrySLAM(SLAMAlgorithm):
    """Fuse wheel prediction with scan-to-scan ICP and accumulate a point map.

    This is a LiDAR-odometry baseline without loop closure. It is intentionally
    distinct from the dead-reckoning baseline and reacts to LiDAR degradation.
    """

    def __init__(
        self,
        wheel_radius_m: float,
        wheel_base_m: float,
        lidar_max_range_m: float,
        max_iterations: int = 15,
        max_correspondence_m: float = 0.6,
        prior_weight: float = 20.0,
        map_stride: int = 3,
    ) -> None:
        self.wheel_radius_m = wheel_radius_m
        self.wheel_base_m = wheel_base_m
        self.lidar_max_range_m = lidar_max_range_m
        self.max_iterations = max_iterations
        self.max_correspondence_m = max_correspondence_m
        self.prior_weight = prior_weight
        self.map_stride = map_stride
        self._pose: Pose2D | None = None
        self._previous_points: np.ndarray | None = None
        self._chunks: list[np.ndarray] = []
        self._updates = 0
        self._fallbacks = 0
        self._last_icp_rmse: float | None = None

    def initialize(self, pose: Pose2D) -> None:
        self._pose = pose
        self._previous_points = None
        self._chunks = []
        self._updates = 0
        self._fallbacks = 0
        self._last_icp_rmse = None

    def _scan_points(self, scan: LidarScan) -> np.ndarray:
        valid = (
            np.isfinite(scan.ranges) & (scan.ranges > 0.02) & (scan.ranges < 0.995 * self.lidar_max_range_m)
        )
        return np.column_stack(
            (scan.ranges[valid] * np.cos(scan.angles[valid]), scan.ranges[valid] * np.sin(scan.angles[valid]))
        )

    def _add_to_map(self, points: np.ndarray) -> None:
        assert self._pose is not None
        selected = points[:: self.map_stride]
        cosine, sine = math.cos(self._pose.yaw), math.sin(self._pose.yaw)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        self._chunks.append(selected @ rotation.T + [self._pose.x, self._pose.y])

    def update(
        self,
        timestamp: float,
        left_rate: float,
        right_rate: float,
        dt_s: float,
        scan: LidarScan,
        imu: IMUSample | None = None,
    ) -> Pose2D:
        del timestamp, imu
        if self._pose is None:
            raise RuntimeError("algorithm must be initialized before update")
        previous_pose = self._pose
        predictor = DifferentialDrive(self.wheel_radius_m, self.wheel_base_m, previous_pose)
        predicted = predictor.step(left_rate, right_rate, dt_s)
        current_points = self._scan_points(scan)
        if self._previous_points is None or len(current_points) < 3 or len(self._previous_points) < 3:
            self._pose = predicted
            if self._previous_points is not None:
                self._fallbacks += 1
        else:
            delta_yaw = float(wrap_angle(predicted.yaw - previous_pose.yaw))
            cosine, sine = math.cos(delta_yaw), math.sin(delta_yaw)
            initial_rotation = np.array([[cosine, -sine], [sine, cosine]])
            displacement = np.array([predicted.x - previous_pose.x, predicted.y - previous_pose.y])
            previous_rotation = np.array(
                [
                    [math.cos(previous_pose.yaw), math.sin(previous_pose.yaw)],
                    [-math.sin(previous_pose.yaw), math.cos(previous_pose.yaw)],
                ]
            )
            initial_translation = previous_rotation @ displacement
            rotation, translation, error, inliers = point_to_line_icp(
                current_points,
                self._previous_points,
                initial_rotation,
                initial_translation,
                self.max_iterations,
                self.max_correspondence_m,
                self.prior_weight,
            )
            if inliers < 3 or not np.isfinite(error):
                self._pose = predicted
                self._fallbacks += 1
            else:
                relative_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
                global_rotation = np.array(
                    [
                        [math.cos(previous_pose.yaw), -math.sin(previous_pose.yaw)],
                        [math.sin(previous_pose.yaw), math.cos(previous_pose.yaw)],
                    ]
                )
                global_translation = global_rotation @ translation
                self._pose = Pose2D(
                    previous_pose.x + float(global_translation[0]),
                    previous_pose.y + float(global_translation[1]),
                    float(wrap_angle(previous_pose.yaw + relative_yaw)),
                )
                self._last_icp_rmse = error
        if len(current_points):
            self._add_to_map(current_points)
        self._previous_points = current_points
        self._updates += 1
        return self._pose

    def reset(self) -> None:
        self._pose = None
        self._previous_points = None
        self._chunks = []
        self._updates = 0
        self._fallbacks = 0

    def get_pose(self) -> Pose2D:
        if self._pose is None:
            raise RuntimeError("algorithm is not initialized")
        return self._pose

    def get_map(self) -> np.ndarray:
        return np.vstack(self._chunks) if self._chunks else np.empty((0, 2), dtype=float)

    def get_diagnostics(self) -> dict[str, float | int | str | None]:
        return {
            "algorithm": "icp_odometry",
            "updates": self._updates,
            "fallbacks": self._fallbacks,
            "last_icp_rmse_m": self._last_icp_rmse,
            "map_points": len(self.get_map()),
        }
