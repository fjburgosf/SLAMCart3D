"""Reachable-frontier exploration over a measured categorical occupancy grid."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from slamcart3d.core.config import NavigationConfig
from slamcart3d.core.models import LidarScan, Pose2D, wrap_angle
from slamcart3d.metrics.occupancy_grid import OccupancyGrid


class FrontierExplorer:
    """Choose observed-free destinations near unknown space, without an oracle.

    Four-connected BFS avoids diagonal corner cutting. Occupied cells are
    inflated by a conservative robot circle. A reactive measured-range guard
    can stop translation; the simulator's separate collision guard is retained.
    Absence of reachable goals does not establish complete world coverage.
    """

    def __init__(self, config: NavigationConfig, radius_m: float, speed_m_s: float, max_range_m: float):
        self.config = config
        self.radius_m = radius_m + config.safety_margin_m
        self.speed_m_s = speed_m_s
        self.max_range_m = max_range_m
        self.grid = OccupancyGrid(config.resolution_m, 1)
        self.route: list[tuple[int, int]] = []
        self.visits: dict[tuple[int, int], int] = {}
        self.steps = 0
        self.replans = 0
        self.reactive_stops = 0
        self.state = "initial_scan"
        self.goal: tuple[float, float] | None = None

    @staticmethod
    def neighbors(cell: tuple[int, int]):
        x, y = cell
        return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

    def _plan(self, pose: Pose2D) -> None:
        resolution = self.grid.resolution_m
        start = self.grid._cell(pose.x, pose.y)
        inflation = math.ceil(self.radius_m / resolution)
        blocked = set()
        for x, y in self.grid.occupied:
            for dx in range(-inflation, inflation + 1):
                for dy in range(-inflation, inflation + 1):
                    if math.hypot(dx, dy) * resolution <= self.radius_m + resolution * 0.71:
                        blocked.add((x + dx, y + dy))
        safe = self.grid.free - blocked
        if start not in safe:
            self.route = []
            self.goal = None
            return
        parents = {start: None}
        queue = deque([start])
        depth = {start: 0}
        best, best_score = None, -math.inf
        observed = self.grid.free | self.grid.occupied
        while queue and len(parents) < 30_000:
            cell = queue.popleft()
            unknown = sum(neighbor not in observed for neighbor in self.neighbors(cell))
            if unknown and depth[cell] >= 2:
                score = 4 * unknown - 0.08 * depth[cell] - 3 * self.visits.get(cell, 0)
                if score > best_score:
                    best, best_score = cell, score
            for neighbor in self.neighbors(cell):
                if neighbor in safe and neighbor not in parents:
                    parents[neighbor] = cell
                    depth[neighbor] = depth[cell] + 1
                    queue.append(neighbor)
        route = []
        while best is not None and best != start:
            route.append(best)
            best = parents[best]
        self.route = route[::-1]
        self.replans += 1
        self.goal = tuple((value + 0.5) * resolution for value in self.route[-1]) if self.route else None

    def command(self, pose: Pose2D, scan: LidarScan | None) -> tuple[float, float]:
        self.steps += 1
        if scan is None:
            self.state = "initial_scan"
            return 0.0, self.config.max_yaw_rate_rad_s
        finite = np.isfinite(scan.ranges) & (scan.ranges > 0.02)
        if np.count_nonzero(finite) < 3:
            self.state = "sensor_unavailable"
            self.route = []
            self.goal = None
            return 0.0, 0.0
        self.grid.observe(scan, pose, self.max_range_m)
        cell = self.grid._cell(pose.x, pose.y)
        self.visits[cell] = self.visits.get(cell, 0) + 1
        front = finite & (np.abs(scan.angles) < 0.55)
        if np.any(front) and float(np.min(scan.ranges[front])) < self.radius_m + 0.1:
            self.reactive_stops += 1
            self.route = []
            self.goal = None
            self.state = "obstacle_turn"
            return 0.0, self.config.max_yaw_rate_rad_s
        if not self.route or self.steps % self.config.replan_steps == 0:
            self._plan(pose)
        if not self.route:
            self.state = "no_reachable_frontier"
            return 0.0, self.config.max_yaw_rate_rad_s
        resolution = self.grid.resolution_m
        while self.route:
            target = np.array([(value + 0.5) * resolution for value in self.route[0]])
            if float(np.linalg.norm(target - [pose.x, pose.y])) > 0.6 * resolution:
                break
            self.route.pop(0)
        if not self.route:
            self.state = "goal_reached"
            return 0.0, self.config.max_yaw_rate_rad_s
        target = [(value + 0.5) * resolution for value in self.route[0]]
        angle = float(wrap_angle(math.atan2(target[1] - pose.y, target[0] - pose.x) - pose.yaw))
        turn = float(np.clip(2.0 * angle, -self.config.max_yaw_rate_rad_s, self.config.max_yaw_rate_rad_s))
        self.state = "following_frontier"
        return (self.speed_m_s * max(0.0, math.cos(angle)) if abs(angle) < 0.55 else 0.0), turn

    def diagnostics(self) -> dict[str, float | int | str | None]:
        return {
            "navigation_policy": "measured_frontier_bfs",
            "navigation_pose_source": "estimated pose; no world geometry",
            "navigation_state": self.state,
            "navigation_replans": self.replans,
            "navigation_reactive_stops": self.reactive_stops,
            "navigation_observed_cells": len(self.grid.free | self.grid.occupied),
            "navigation_goal_x_m": self.goal[0] if self.goal else None,
            "navigation_goal_y_m": self.goal[1] if self.goal else None,
        }
