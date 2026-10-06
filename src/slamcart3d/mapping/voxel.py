"""Sparse ray-carved 3D occupancy, independent of the simulated world."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class SurfaceMesh:
    """Boundary triangles of measured occupied cells, not a watertight scene."""

    vertices: np.ndarray
    triangles: np.ndarray


class VoxelMap:
    """Categorical evidence: endpoints occupied, ray interiors free, rest unknown.

    Counts are supporting observations, not calibrated probabilities. A voxel
    needs min_hits to enter the reconstructed surface. No-hit or missing rays
    are not supplied here and never clear unknown space. One scan contributes
    at most one hit and one free observation per cell. Occupied evidence wins
    over free evidence; dynamic-object removal is intentionally not claimed.
    """

    def __init__(self, resolution_m: float = 0.25, min_hits: int = 2):
        if not np.isfinite(resolution_m) or not 0.05 <= resolution_m <= 1.0:
            raise ValueError("voxel resolution must be between 0.05 and 1.0 m")
        if type(min_hits) is not int or min_hits < 1:
            raise ValueError("voxel min_hits must be a positive integer")
        self.resolution_m = float(resolution_m)
        self.min_hits = min_hits
        self.hits: dict[tuple[int, int, int], int] = {}
        self.free: set[tuple[int, int, int]] = set()

    @property
    def occupied(self) -> set[tuple[int, int, int]]:
        return {cell for cell, hits in self.hits.items() if hits >= self.min_hits}

    def observe(self, origin: np.ndarray, points: np.ndarray) -> None:
        origin, points = np.asarray(origin, dtype=float), np.asarray(points, dtype=float)
        if origin.shape != (3,) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("voxel observation requires XYZ origin and N x 3 endpoints")
        if not np.isfinite(origin).all() or not np.isfinite(points).all():
            raise ValueError("voxel observation must be finite")
        hit_cells = set()
        free_cells = set()
        for point in points:
            direction = point - origin
            count = max(1, int(np.ceil(np.max(np.abs(direction)) / self.resolution_m * 2)))
            # Half-cell oversampling visits crossed ray cells; remove endpoints
            # explicitly so a hit cannot clear its own cell.
            samples = origin + np.arange(count)[:, None] / count * direction
            cells = np.floor(samples / self.resolution_m).astype(int)
            free_cells.update(map(tuple, cells.tolist()))
            hit_cells.add(tuple(np.floor(point / self.resolution_m).astype(int).tolist()))
        free_cells -= hit_cells
        self.free.update(free_cells)
        for cell in hit_cells:
            self.hits[cell] = self.hits.get(cell, 0) + 1
        self.free.difference_update(self.hits)

    def rows(self):
        for cell in sorted(self.free | self.hits.keys()):
            hits = self.hits.get(cell, 0)
            state = "occupied" if hits >= self.min_hits else "candidate" if hits else "free"
            yield (*cell, *((np.asarray(cell) + 0.5) * self.resolution_m), state, hits)

    def surface(self) -> SurfaceMesh:
        """Extract outward faces, suppressing internal faces between neighbors."""
        vertices = []
        triangles = []
        index = {}
        occupied = self.occupied
        for cell in sorted(occupied):
            for axis, sign in product(range(3), (-1, 1)):
                neighbor = list(cell)
                neighbor[axis] += sign
                if tuple(neighbor) in occupied:
                    continue
                other = [value for value in range(3) if value != axis]
                face = []
                for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    corner = list(cell)
                    corner[axis] += int(sign > 0)
                    corner[other[0]] += a
                    corner[other[1]] += b
                    key = tuple(corner)
                    if key not in index:
                        index[key] = len(vertices)
                        vertices.append(np.asarray(corner) * self.resolution_m)
                    face.append(index[key])
                # Cross product orientation differs for the Y face.
                if sign * (-1 if axis == 1 else 1) < 0:
                    face.reverse()
                triangles.extend(((face[0], face[1], face[2]), (face[0], face[2], face[3])))
        return SurfaceMesh(
            np.asarray(vertices, dtype=float).reshape(-1, 3),
            np.asarray(triangles, dtype=int).reshape(-1, 3),
        )


def pose_matrix(state: np.ndarray) -> np.ndarray:
    """Robot-to-world SE(3), Euler convention Rz(yaw) Ry(pitch) Rx(roll)."""
    x, y, z, roll, pitch, yaw = state
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    matrix[:3, 3] = [x, y, z]
    return matrix


def transform_points(points: np.ndarray, state: np.ndarray) -> np.ndarray:
    matrix = pose_matrix(state)
    return points @ matrix[:3, :3].T + matrix[:3, 3]
