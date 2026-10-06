"""Conservative scan-verified SE(2) loop constraints and final pose optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from slamcart3d.core.models import IMUSample, LidarScan, Pose2D, wrap_angle
from slamcart3d.slam.adaptive_hybrid import AdaptiveHybridSLAM
from slamcart3d.slam.icp_odometry import point_to_line_icp


def rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def relative_pose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.r_[rotation(first[2]).T @ (second[:2] - first[:2]), wrap_angle(second[2] - first[2])]


@dataclass(frozen=True)
class PoseEdge:
    first: int
    second: int
    measurement: np.ndarray
    information: np.ndarray
    is_loop: bool = False


def graph_cost(poses: np.ndarray, edges: list[PoseEdge]) -> float:
    total = 0.0
    for edge in edges:
        residual = relative_pose(poses[edge.first], poses[edge.second]) - edge.measurement
        residual[2] = float(wrap_angle(residual[2]))
        # Bounded influence of inconsistent edges, in standardized units.
        norm = math.sqrt(float(np.sum(edge.information * residual**2)))
        total += norm**2 if norm <= 3.0 else 6.0 * norm - 9.0
    return total


def optimize_pose_graph(poses: np.ndarray, edges: list[PoseEdge], iterations: int = 15) -> np.ndarray:
    """Anchored robust Gauss-Newton with a cost-decreasing line search."""

    state = np.asarray(poses, dtype=float).copy()
    if len(state) < 2 or not edges:
        return state
    size = 3 * (len(state) - 1)
    for _ in range(iterations):
        hessian, gradient = np.zeros((size, size)), np.zeros(size)
        for edge in edges:
            i, j = edge.first, edge.second
            predicted = relative_pose(state[i], state[j])
            residual = predicted - edge.measurement
            residual[2] = float(wrap_angle(residual[2]))
            rot = rotation(state[i, 2]).T
            first = np.zeros((3, 3))
            first[:2, :2] = -rot
            first[:2, 2] = (predicted[1], -predicted[0])
            first[2, 2] = -1
            second = np.zeros((3, 3))
            second[:2, :2] = rot
            second[2, 2] = 1
            norm = math.sqrt(float(np.sum(edge.information * residual**2)))
            weight = edge.information * min(1.0, 3.0 / max(norm, 1e-12))
            blocks = [(node, jac) for node, jac in ((i, first), (j, second)) if node > 0]
            for node, jac in blocks:
                block = slice(3 * (node - 1), 3 * node)
                gradient[block] += jac.T @ (weight * residual)
                for other, other_jac in blocks:
                    target = slice(3 * (other - 1), 3 * other)
                    hessian[block, target] += jac.T @ (weight[:, None] * other_jac)
        delta = np.linalg.solve(hessian + 1e-6 * np.eye(size), -gradient).reshape(-1, 3)
        cost = graph_cost(state, edges)
        accepted = False
        for factor in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = state.copy()
            candidate[1:] += factor * delta
            candidate[:, 2] = wrap_angle(candidate[:, 2])
            if graph_cost(candidate, edges) <= cost + 1e-12:
                state, accepted = candidate, True
                break
        if not accepted or float(np.linalg.norm(delta)) < 1e-6:
            break
    return state


class PoseGraphHybridSLAM(AdaptiveHybridSLAM):
    """Keep the online hybrid estimate; optimize verified loops at run completion.

    Candidates require spatial proximity, time separation, similar orientation,
    non-collinear scan geometry, and high bidirectional scan overlap. These
    conservative gates can miss loops; repeated geometry can still cause false
    matches. No true pose, world walls, or labelled training data are consumed.
    """

    def initialize(self, pose: Pose2D) -> None:
        super().initialize(pose)
        self.history = [pose.as_array().copy()]
        self.local_scans: list[np.ndarray] = []
        self.map_scans: list[np.ndarray] = []
        self.key_indices = [0]
        self.key_poses = [pose.as_array().copy()]
        self.key_scans = [np.empty((0, 2))]
        self.edges: list[PoseEdge] = []
        self.loop_pairs: list[tuple[int, int]] = []
        self.candidates = 0
        self.last_loop_step = -100
        self.cost_before = self.cost_after = 0.0

    def update(
        self,
        timestamp: float,
        left_rate: float,
        right_rate: float,
        dt_s: float,
        scan: LidarScan,
        imu: IMUSample | None = None,
    ) -> Pose2D:
        before = len(self._chunks)
        pose = super().update(timestamp, left_rate, right_rate, dt_s, scan, imu)
        self.map_scans.append(
            self._scan_points(scan)[::2] if len(self._chunks) > before else np.empty((0, 2))
        )
        valid = (
            np.isfinite(scan.ranges) & (scan.ranges > 0.02) & (scan.ranges < 0.995 * self.lidar_max_range_m)
        )
        points = np.column_stack(
            (scan.ranges[valid] * np.cos(scan.angles[valid]), scan.ranges[valid] * np.sin(scan.angles[valid]))
        )
        points = points[:: max(1, math.ceil(len(points) / 180))]
        self.local_scans.append(points)
        self.history.append(pose.as_array().copy())
        step = len(self.history) - 1
        if (step == 1 or step % 10 == 0) and len(self.key_poses) < 120:
            self._add_keyframe(step, pose.as_array(), points)
        return pose

    def _add_keyframe(self, step: int, pose: np.ndarray, points: np.ndarray) -> None:
        node = len(self.key_poses)
        self.edges.append(
            PoseEdge(node - 1, node, relative_pose(self.key_poses[-1], pose), np.array([400.0, 400.0, 400.0]))
        )
        if len(points) >= 12 and step - self.last_loop_step >= 50:
            covariance = np.cov(points.T)
            if float(np.linalg.eigvalsh(covariance)[0]) > 0.01:
                for old in range(1, node):
                    prior = self.key_poses[old]
                    travel = float(
                        np.sum(
                            np.linalg.norm(
                                np.diff(np.asarray(self.history[self.key_indices[old] :]), axis=0)[:, :2],
                                axis=1,
                            )
                        )
                    )
                    if (
                        travel < 2.0
                        or step - self.key_indices[old] < 80
                        or len(self.key_scans[old]) < 12
                        or np.linalg.norm(prior[:2] - pose[:2]) > 0.8
                        or abs(float(wrap_angle(prior[2] - pose[2]))) > 0.5
                    ):
                        continue
                    self.candidates += 1
                    target = self.key_scans[old] @ rotation(prior[2]).T + prior[:2]
                    rot, translation, rmse, inliers = point_to_line_icp(
                        points, target, rotation(pose[2]), pose[:2], 20, 0.5, 0.2
                    )
                    fraction = inliers / len(points)
                    if fraction < 0.75 or rmse > 0.09:
                        continue
                    candidate = np.r_[translation, math.atan2(rot[1, 0], rot[0, 0])]
                    corrected = points @ rotation(candidate[2]).T + candidate[:2]
                    distances = np.sqrt(
                        np.min(np.sum((target[:, None] - corrected[None]) ** 2, axis=2), axis=1)
                    )
                    if (
                        float(np.mean(distances < 0.35)) < 0.75
                        or np.linalg.norm(candidate[:2] - pose[:2]) > 0.6
                        or abs(float(wrap_angle(candidate[2] - pose[2]))) > 0.3
                    ):
                        continue
                    self.edges.append(
                        PoseEdge(
                            old,
                            node,
                            relative_pose(prior, candidate),
                            np.array([1111.0, 1111.0, 625.0]),
                            True,
                        )
                    )
                    self.loop_pairs.append((self.key_indices[old], step))
                    self.last_loop_step = step
                    break
        self.key_indices.append(step)
        self.key_poses.append(pose.copy())
        self.key_scans.append(points.copy())

    def finalize(self) -> np.ndarray:
        online = np.asarray(self.history)
        if self.key_indices[-1] != len(online) - 1 and len(self.key_poses) < 120:
            self._add_keyframe(len(online) - 1, online[-1], self.local_scans[-1])
        poses = np.asarray(self.key_poses)
        self.cost_before = graph_cost(poses, self.edges)
        optimized = optimize_pose_graph(poses, self.edges) if self.loop_pairs else poses.copy()
        self.cost_after = graph_cost(optimized, self.edges)
        angles = wrap_angle(optimized[:, 2] - poses[:, 2])
        shifts = np.array(
            [new[:2] - rotation(a) @ old[:2] for new, old, a in zip(optimized, poses, angles, strict=True)]
        )
        corrected = online.copy()
        steps = np.arange(len(online))
        interpolated_angle = np.interp(steps, self.key_indices, np.unwrap(angles))
        x = np.interp(steps, self.key_indices, shifts[:, 0])
        y = np.interp(steps, self.key_indices, shifts[:, 1])
        for index, angle in enumerate(interpolated_angle):
            corrected[index, :2] = rotation(float(angle)) @ online[index, :2] + [x[index], y[index]]
            corrected[index, 2] = float(wrap_angle(online[index, 2] + angle))
        self._chunks = [
            points @ rotation(pose[2]).T + pose[:2]
            for points, pose in zip(self.map_scans, corrected[1:], strict=True)
            if len(points)
        ]
        self._pose = Pose2D(*corrected[-1])
        return corrected

    def get_diagnostics(self) -> dict[str, float | int | str | None]:
        result = super().get_diagnostics()
        result.update(
            algorithm="posegraph_hybrid",
            loop_candidates=self.candidates,
            accepted_loops=len(self.loop_pairs),
            posegraph_keyframes=len(self.key_poses),
            posegraph_cost_before=self.cost_before,
            posegraph_cost_after=self.cost_after,
            loop_optimization="at run completion; online trajectory retained",
        )
        return result

    def reset(self) -> None:
        self.initialize(Pose2D(0.0, 0.0, 0.0))
        self._pose = None
