"""Dependency-light 3D world representation with planar ray casting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WallSegment:
    """A vertical wall represented by its ground-plane endpoints and height."""

    x1: float
    y1: float
    x2: float
    y2: float
    height_m: float = 2.5


@dataclass(frozen=True)
class MovingBox:
    """A box following a bounded, deterministic horizontal oscillation."""

    x_m: float
    y_m: float
    width_m: float
    depth_m: float
    amplitude_x_m: float
    frequency_hz: float
    phase_rad: float
    height_m: float = 1.6

    def segments_at(self, timestamp_s: float) -> tuple[WallSegment, ...]:
        x = self.x_m + self.amplitude_x_m * math.sin(
            2.0 * math.pi * self.frequency_hz * timestamp_s + self.phase_rad
        )
        return tuple(_box(x, self.y_m, self.width_m, self.depth_m, self.height_m))


@dataclass(frozen=True)
class World3D:
    """A collection of vertical surfaces embedded in three dimensions."""

    name: str
    segments: tuple[WallSegment, ...]
    moving_boxes: tuple[MovingBox, ...] = ()

    def segments_at(self, timestamp_s: float) -> tuple[WallSegment, ...]:
        """Return static walls plus movable surfaces at a simulation time."""

        return self.segments + tuple(
            segment for moving in self.moving_boxes for segment in moving.segments_at(timestamp_s)
        )

    def path_clearance_m(
        self, start: tuple[float, float], end: tuple[float, float], timestamp_s: float
    ) -> float:
        """Minimum planar distance from a straight path chord to any wall."""

        path_a = np.asarray(start, dtype=float)
        path_b = np.asarray(end, dtype=float)

        def point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            direction = b - a
            length_sq = float(np.dot(direction, direction))
            fraction = (
                float(np.clip(np.dot(point - a, direction) / length_sq, 0.0, 1.0)) if length_sq else 0.0
            )
            return float(np.linalg.norm(point - a - fraction * direction))

        def cross(a: np.ndarray, b: np.ndarray) -> float:
            return float(a[0] * b[1] - a[1] * b[0])

        nearest = math.inf
        for wall in self.segments_at(timestamp_s):
            wall_a = np.array([wall.x1, wall.y1])
            wall_b = np.array([wall.x2, wall.y2])
            path_direction = path_b - path_a
            wall_direction = wall_b - wall_a
            denominator = cross(path_direction, wall_direction)
            if abs(denominator) > 1e-12:
                offset = wall_a - path_a
                path_fraction = cross(offset, wall_direction) / denominator
                wall_fraction = cross(offset, path_direction) / denominator
                if 0.0 <= path_fraction <= 1.0 and 0.0 <= wall_fraction <= 1.0:
                    return 0.0
            nearest = min(
                nearest,
                point_segment_distance(path_a, wall_a, wall_b),
                point_segment_distance(path_b, wall_a, wall_b),
                point_segment_distance(wall_a, path_a, path_b),
                point_segment_distance(wall_b, path_a, path_b),
            )
        return nearest

    def ray_distance(
        self, x: float, y: float, angle_rad: float, max_range_m: float, timestamp_s: float = 0.0
    ) -> float:
        """Return distance to the closest wall intersected by a planar ray."""

        ray = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)
        origin = np.array([x, y], dtype=float)
        nearest = max_range_m
        for wall in self.segments_at(timestamp_s):
            start = np.array([wall.x1, wall.y1], dtype=float)
            segment = np.array([wall.x2 - wall.x1, wall.y2 - wall.y1], dtype=float)
            denominator = ray[0] * segment[1] - ray[1] * segment[0]
            if abs(denominator) < 1e-12:
                continue
            offset = start - origin
            distance = (offset[0] * segment[1] - offset[1] * segment[0]) / denominator
            fraction = (offset[0] * ray[1] - offset[1] * ray[0]) / denominator
            if distance >= 0.0 and 0.0 <= fraction <= 1.0:
                nearest = min(nearest, float(distance))
        return nearest

    def ray_distance_3d(
        self,
        x: float,
        y: float,
        z: float,
        azimuth_rad: float,
        elevation_rad: float,
        max_range_m: float,
        timestamp_s: float = 0.0,
    ) -> float:
        """Cast a 3D ray against finite-height walls and the flat ground plane."""

        if z < 0.0:
            raise ValueError("ray origin must not be below the ground plane")
        horizontal_fraction = math.cos(elevation_rad)
        vertical_fraction = math.sin(elevation_rad)
        nearest = max_range_m
        if vertical_fraction < -1e-12:
            nearest = min(nearest, -z / vertical_fraction)
        if horizontal_fraction <= 1e-12:
            return nearest
        ray = np.array([math.cos(azimuth_rad), math.sin(azimuth_rad)], dtype=float)
        origin = np.array([x, y], dtype=float)
        for wall in self.segments_at(timestamp_s):
            start = np.array([wall.x1, wall.y1], dtype=float)
            segment = np.array([wall.x2 - wall.x1, wall.y2 - wall.y1], dtype=float)
            denominator = ray[0] * segment[1] - ray[1] * segment[0]
            if abs(denominator) < 1e-12:
                continue
            offset = start - origin
            horizontal_distance = (offset[0] * segment[1] - offset[1] * segment[0]) / denominator
            fraction = (offset[0] * ray[1] - offset[1] * ray[0]) / denominator
            if horizontal_distance < 0.0 or not 0.0 <= fraction <= 1.0:
                continue
            slant_range = horizontal_distance / horizontal_fraction
            height = z + slant_range * vertical_fraction
            if 0.0 <= height <= wall.height_m:
                nearest = min(nearest, float(slant_range))
        return nearest


def _box(x: float, y: float, width: float, depth: float, height: float = 1.2) -> list[WallSegment]:
    return [
        WallSegment(x, y, x + width, y, height),
        WallSegment(x + width, y, x + width, y + depth, height),
        WallSegment(x + width, y + depth, x, y + depth, height),
        WallSegment(x, y + depth, x, y, height),
    ]


def indoor_simple(seed: int) -> World3D:
    """Generate a reproducible 10 m by 10 m room with seeded static objects."""

    rng = np.random.default_rng(seed)
    walls = [
        WallSegment(0.0, 0.0, 10.0, 0.0),
        WallSegment(10.0, 0.0, 10.0, 10.0),
        WallSegment(10.0, 10.0, 0.0, 10.0),
        WallSegment(0.0, 10.0, 0.0, 0.0),
        WallSegment(2.0, 2.0, 4.0, 2.0),
        WallSegment(7.0, 1.0, 7.0, 3.0),
    ]
    for anchor_x, anchor_y in ((1.0, 7.5), (7.5, 7.0)):
        width, depth = rng.uniform(0.5, 1.0, size=2)
        walls.extend(_box(anchor_x, anchor_y, float(width), float(depth)))
    return World3D(name="indoor_simple", segments=tuple(walls))


def indoor_corridor(seed: int) -> World3D:
    """Generate a long corridor with seeded wall recesses and side alcoves."""

    rng = np.random.default_rng(seed)
    upper_recess = float(rng.uniform(0.6, 1.2))
    lower_recess = float(rng.uniform(0.6, 1.2))
    walls = [
        WallSegment(0.0, 0.0, 14.0, 0.0),
        WallSegment(14.0, 0.0, 14.0, 10.0),
        WallSegment(14.0, 10.0, 0.0, 10.0),
        WallSegment(0.0, 10.0, 0.0, 0.0),
        WallSegment(0.0, 2.0, 3.0, 2.0),
        WallSegment(3.0, 2.0, 3.0, 2.0 - lower_recess),
        WallSegment(4.0, 2.0, 14.0, 2.0),
        WallSegment(0.0, 8.8, 10.0, 8.8),
        WallSegment(10.0, 8.8, 10.0, 8.8 + upper_recess),
        WallSegment(11.0, 8.8, 14.0, 8.8),
    ]
    return World3D(name="indoor_corridor", segments=tuple(walls))


def indoor_complex(seed: int) -> World3D:
    """Generate connected rooms above and below a feature-repetitive corridor."""

    rng = np.random.default_rng(seed)
    walls = [
        WallSegment(0.0, 0.0, 16.0, 0.0),
        WallSegment(16.0, 0.0, 16.0, 12.0),
        WallSegment(16.0, 12.0, 0.0, 12.0),
        WallSegment(0.0, 12.0, 0.0, 0.0),
        # The central corridor spans y=3..5. Its door gaps remain open.
        WallSegment(0.0, 3.0, 2.0, 3.0),
        WallSegment(3.0, 3.0, 9.0, 3.0),
        WallSegment(10.0, 3.0, 16.0, 3.0),
        WallSegment(0.0, 5.0, 5.0, 5.0),
        WallSegment(6.0, 5.0, 12.0, 5.0),
        WallSegment(13.0, 5.0, 16.0, 5.0),
        # Lower and upper partitions create distinct rooms, each with a gap.
        WallSegment(8.0, 0.0, 8.0, 1.2),
        WallSegment(8.0, 2.2, 8.0, 3.0),
        WallSegment(8.0, 5.0, 8.0, 7.0),
        WallSegment(8.0, 8.0, 8.0, 12.0),
        WallSegment(12.0, 5.0, 12.0, 9.0),
        WallSegment(12.0, 10.0, 12.0, 12.0),
    ]
    for x, y in ((1.0, 0.8), (10.0, 0.8), (1.0, 7.0), (9.0, 9.2), (13.4, 7.0)):
        width = float(rng.uniform(0.55, 0.85))
        depth = float(rng.uniform(0.5, 0.8))
        walls.extend(_box(x, y, width, depth, height=1.4))
    return World3D(name="indoor_complex", segments=tuple(walls))


def warehouse_shelves(seed: int) -> World3D:
    """Generate a room with four seeded, non-overlapping storage racks."""

    rng = np.random.default_rng(seed)
    walls = [
        WallSegment(0.0, 0.0, 14.0, 0.0),
        WallSegment(14.0, 0.0, 14.0, 12.0),
        WallSegment(14.0, 12.0, 0.0, 12.0),
        WallSegment(0.0, 12.0, 0.0, 0.0),
    ]
    for x, y in ((1.0, 2.0), (1.0, 7.0), (9.0, 2.0), (9.0, 7.0)):
        width = float(rng.uniform(1.0, 1.4))
        depth = float(rng.uniform(1.7, 2.3))
        walls.extend(_box(x, y, width, depth, height=1.8))
    return World3D(name="warehouse_shelves", segments=tuple(walls))


def indoor_maze(seed: int) -> World3D:
    """Generate a connected, closed 15 x 12 m maze with 3 m wide passages.

    Randomized depth-first carving creates a spanning tree of 5 x 4 cells:
    every cell is reachable, with turns and dead ends but no enclosed islands.
    The seed controls topology, not the exploration route. Default (5, 4)
    lies inside a cell, clear of any possible wall. Walls are 2.5 m high;
    the outer boundary stays closed for controlled mapping experiments.
    """

    rng = np.random.default_rng(seed)
    columns, rows, cell_size = 5, 4, 3.0
    visited = {(1, 1)}
    stack = [(1, 1)]
    openings = set()
    while stack:
        x, y = stack[-1]
        candidates = [
            (nx, ny)
            for nx, ny in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))
            if 0 <= nx < columns and 0 <= ny < rows and (nx, ny) not in visited
        ]
        if not candidates:
            stack.pop()
            continue
        neighbor = candidates[int(rng.integers(len(candidates)))]
        openings.add(frozenset(((x, y), neighbor)))
        visited.add(neighbor)
        stack.append(neighbor)

    walls = _box(0.0, 0.0, columns * cell_size, rows * cell_size, height=2.5)
    # Emit each shared cell boundary once, unless carving opened it.
    for y in range(rows):
        for x in range(columns):
            if x + 1 < columns and frozenset(((x, y), (x + 1, y))) not in openings:
                boundary = (x + 1) * cell_size
                walls.append(WallSegment(boundary, y * cell_size, boundary, (y + 1) * cell_size))
            if y + 1 < rows and frozenset(((x, y), (x, y + 1))) not in openings:
                boundary = (y + 1) * cell_size
                walls.append(WallSegment(x * cell_size, boundary, (x + 1) * cell_size, boundary))
    return World3D(name="indoor_maze", segments=tuple(walls))


def indoor_dynamic(seed: int) -> World3D:
    """Add one seeded moving obstacle to the static indoor room."""

    static = indoor_simple(seed)
    rng = np.random.default_rng(seed + 13)
    moving = MovingBox(
        x_m=1.5,
        y_m=5.0,
        width_m=0.6,
        depth_m=0.6,
        amplitude_x_m=0.45,
        frequency_hz=0.2,
        phase_rad=float(rng.uniform(0.0, 2.0 * math.pi)),
    )
    return World3D(name="indoor_dynamic", segments=static.segments, moving_boxes=(moving,))


def outdoor_blocks(seed: int) -> World3D:
    """Generate sparse building footprints on an otherwise open, flat plane."""

    rng = np.random.default_rng(seed)
    walls: list[WallSegment] = []
    for x, y in ((0.0, 0.0), (9.0, 0.0), (0.0, 9.0), (9.0, 9.0)):
        width = float(rng.uniform(1.4, 2.0))
        depth = float(rng.uniform(1.4, 2.0))
        height = float(rng.uniform(2.0, 3.0))
        walls.extend(_box(x, y, width, depth, height))
    return World3D(name="outdoor_blocks", segments=tuple(walls))


def make_world(name: str, seed: int) -> World3D:
    """Construct a named world, failing explicitly for unsupported backends."""

    worlds = {
        "indoor_simple": indoor_simple,
        "indoor_corridor": indoor_corridor,
        "indoor_complex": indoor_complex,
        "warehouse_shelves": warehouse_shelves,
        "indoor_maze": indoor_maze,
        "indoor_dynamic": indoor_dynamic,
        "outdoor_blocks": outdoor_blocks,
    }
    try:
        return worlds[name](seed)
    except KeyError as exc:
        raise ValueError(f"unsupported world: {name}") from exc
