"""Geometric validity diagnostics for sampled ground-truth motion."""

from __future__ import annotations

import numpy as np

from slamcart3d.worlds.geometry import World3D


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _intersects(first: np.ndarray, second: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    """Check finite 2D segment contact, including endpoints and overlap."""

    movement = second - first
    wall = end - start
    if float(movement @ movement) <= 1e-24:
        wall_length_squared = float(wall @ wall)
        if wall_length_squared <= 1e-24:
            return bool(np.linalg.norm(first - start) <= 1e-12)
        fraction = float((first - start) @ wall) / wall_length_squared
        return abs(_cross(first - start, wall)) <= 1e-12 and -1e-12 <= fraction <= 1.0 + 1e-12
    denominator = _cross(movement, wall)
    offset = start - first
    if abs(denominator) <= 1e-12:
        if abs(_cross(offset, movement)) > 1e-12:
            return False
        return bool(
            np.all(np.maximum(first, second) >= np.minimum(start, end) - 1e-12)
            and np.all(np.maximum(start, end) >= np.minimum(first, second) - 1e-12)
        )
    movement_fraction = _cross(offset, wall) / denominator
    wall_fraction = _cross(offset, movement) / denominator
    return -1e-12 <= movement_fraction <= 1.0 + 1e-12 and -1e-12 <= wall_fraction <= 1.0 + 1e-12


def static_wall_contact_steps(ground_truth: np.ndarray, world: World3D) -> int:
    """Count consecutive pose chords intersecting any static wall segment.

    The robot is represented by its center point and consecutive samples by
    straight chords. This diagnostic does not model body radius, continuous
    curved-path collisions, moving obstacles, or a collision response.
    """

    if ground_truth.ndim != 2 or ground_truth.shape[1] < 2:
        raise ValueError("ground_truth must be an N x 2-or-more array")
    if not np.isfinite(ground_truth[:, :2]).all():
        raise ValueError("ground_truth positions must be finite")
    walls = [(np.array([wall.x1, wall.y1]), np.array([wall.x2, wall.y2])) for wall in world.segments]
    count = 0
    for first, second in zip(ground_truth[:-1, :2], ground_truth[1:, :2], strict=True):
        count += any(_intersects(first, second, start, end) for start, end in walls)
    return count
