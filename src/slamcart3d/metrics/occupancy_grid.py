"""Sparse observed occupancy grid and objective static-wall evaluation."""

from __future__ import annotations

import math

from slamcart3d.core.models import LidarScan, Pose2D
from slamcart3d.worlds.geometry import World3D

GridCell = tuple[int, int]


class OccupancyGrid:
    """Map valid LiDAR hits and the ray interiors using estimated planar poses.

    Occupied evidence takes precedence over free evidence. Missing/no-hit beams
    are ignored, so unobserved cells remain unknown rather than becoming free.
    This is a deterministic categorical map, not a calibrated log-odds model.
    """

    def __init__(self, resolution_m: float = 0.2, beam_stride: int = 5) -> None:
        if not math.isfinite(resolution_m) or not 0.05 <= resolution_m <= 1.0:
            raise ValueError("occupancy grid resolution must be between 0.05 and 1.0 m")
        if not isinstance(beam_stride, int) or beam_stride < 1:
            raise ValueError("occupancy grid beam stride must be a positive integer")
        self.resolution_m = resolution_m
        self.beam_stride = beam_stride
        self.occupied: set[GridCell] = set()
        self.free: set[GridCell] = set()

    def _cell(self, x_m: float, y_m: float) -> GridCell:
        return math.floor(x_m / self.resolution_m), math.floor(y_m / self.resolution_m)

    def _touching_cells(self, x_m: float, y_m: float) -> set[GridCell]:
        """Include both sides when a wall lies exactly on a cell boundary."""

        scaled_x, scaled_y = x_m / self.resolution_m, y_m / self.resolution_m
        columns = {math.floor(scaled_x)}
        rows = {math.floor(scaled_y)}
        if math.isclose(scaled_x, round(scaled_x), rel_tol=0.0, abs_tol=1e-9):
            columns.add(round(scaled_x) - 1)
        if math.isclose(scaled_y, round(scaled_y), rel_tol=0.0, abs_tol=1e-9):
            rows.add(round(scaled_y) - 1)
        return {(column, row) for column in columns for row in rows}

    def observe(self, scan: LidarScan, pose: Pose2D, max_range_m: float) -> None:
        """Add finite hit endpoints and free ray interiors from one scan."""

        if not math.isfinite(max_range_m) or max_range_m <= 0.0:
            raise ValueError("max_range_m must be finite and positive")
        if scan.angles.shape != scan.ranges.shape:
            raise ValueError("scan angles and ranges must have the same shape")
        for angle, measured_range in zip(
            scan.angles[:: self.beam_stride], scan.ranges[:: self.beam_stride], strict=True
        ):
            distance = float(measured_range)
            if not math.isfinite(distance) or not 0.02 < distance < 0.995 * max_range_m:
                continue
            heading = pose.yaw + float(angle)
            dx, dy = math.cos(heading), math.sin(heading)
            endpoint = self._cell(pose.x + distance * dx, pose.y + distance * dy)
            # Keep the terminal wall-touching cell out of free evidence even
            # when the surface coincides with a grid boundary.
            free_distance = max(0.0, distance - 1.5 * self.resolution_m)
            if free_distance > 0.0:
                steps = max(1, math.ceil(2.0 * free_distance / self.resolution_m))
                if steps > 10_000:
                    raise ValueError("occupancy ray exceeds the 10000-sample safety limit")
                for step in range(steps + 1):
                    along = free_distance * step / steps
                    cell = self._cell(pose.x + along * dx, pose.y + along * dy)
                    if cell not in self.occupied:
                        self.free.add(cell)
            self.occupied.add(endpoint)
            self.free.discard(endpoint)

    def reference_occupied(self, world: World3D) -> set[GridCell]:
        """Rasterize static wall segments at quarter-cell spacing."""

        cells: set[GridCell] = set()
        for wall in world.segments:
            length = math.hypot(wall.x2 - wall.x1, wall.y2 - wall.y1)
            steps = max(1, math.ceil(4.0 * length / self.resolution_m))
            for step in range(steps + 1):
                fraction = step / steps
                cells.update(
                    self._touching_cells(
                        wall.x1 + fraction * (wall.x2 - wall.x1),
                        wall.y1 + fraction * (wall.y2 - wall.y1),
                    )
                )
        return cells

    def metrics(self, world: World3D) -> dict[str, float]:
        """Compare occupied coverage globally and free correctness where observed."""

        reference = self.reference_occupied(world)
        if not reference:
            raise ValueError("occupancy evaluation requires static walls")
        true_positive = len(self.occupied & reference)
        false_positive = len(self.occupied - reference)
        false_negative = len(reference - self.occupied)
        precision = true_positive / (true_positive + false_positive) if self.occupied else 0.0
        recall = true_positive / len(reference)
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou = true_positive / (true_positive + false_positive + false_negative)
        free_accuracy = len(self.free - reference) / len(self.free) if self.free else 0.0
        return {
            "occupancy_occupied_precision": precision,
            "occupancy_occupied_recall": recall,
            "occupancy_occupied_f1": f1,
            "occupancy_occupied_iou": iou,
            "occupancy_observed_free_accuracy": free_accuracy,
            "occupancy_occupied_cells": float(len(self.occupied)),
            "occupancy_observed_free_cells": float(len(self.free)),
            "occupancy_reference_occupied_cells": float(len(reference)),
        }

    def rows(self, world: World3D) -> list[tuple[int, int, float, float, str, str]]:
        """List observed and reference-wall cells, retaining unknown predictions."""

        reference = self.reference_occupied(world)
        rows = []
        for x, y in sorted(self.occupied | self.free | reference):
            cell = (x, y)
            predicted = "occupied" if cell in self.occupied else "free" if cell in self.free else "unknown"
            rows.append(
                (
                    x,
                    y,
                    (x + 0.5) * self.resolution_m,
                    (y + 0.5) * self.resolution_m,
                    predicted,
                    "occupied" if cell in reference else "free",
                )
            )
        return rows
