"""Wall-surface quality of an estimated-pose multilayer LiDAR map."""

from __future__ import annotations

import math

import numpy as np

from slamcart3d.metrics.map_quality import _directed_nearest_distances
from slamcart3d.worlds.geometry import World3D

WALL_SURFACE_SAMPLE_SPACING_M = 0.2
WALL_3D_MATCH_TOLERANCE_M = 0.25
GROUND_EXCLUSION_HEIGHT_M = 0.1


def has_evaluable_wall_surface(world: World3D) -> bool:
    """Whether the default ground exclusion leaves a static-wall reference."""

    return any(wall.height_m > GROUND_EXCLUSION_HEIGHT_M + 1e-9 for wall in world.segments)


def sample_wall_surface_points(
    world: World3D,
    spacing_m: float = WALL_SURFACE_SAMPLE_SPACING_M,
    ground_exclusion_height_m: float = GROUND_EXCLUSION_HEIGHT_M,
) -> np.ndarray:
    """Sample static wall surfaces above the excluded ground-adjacent band."""

    if spacing_m <= 0.0 or not math.isfinite(spacing_m):
        raise ValueError("wall-surface spacing must be finite and positive")
    if ground_exclusion_height_m < 0.0 or not math.isfinite(ground_exclusion_height_m):
        raise ValueError("ground exclusion height must be finite and nonnegative")
    chunks: list[np.ndarray] = []
    for wall in world.segments:
        lower_height = ground_exclusion_height_m + 1e-9
        if wall.height_m <= lower_height:
            continue
        start = np.array([wall.x1, wall.y1], dtype=float)
        delta = np.array([wall.x2 - wall.x1, wall.y2 - wall.y1], dtype=float)
        horizontal_steps = max(1, math.ceil(float(np.linalg.norm(delta)) / spacing_m))
        vertical_steps = max(1, math.ceil((wall.height_m - lower_height) / spacing_m))
        along, heights = np.meshgrid(
            np.linspace(0.0, 1.0, horizontal_steps + 1),
            np.linspace(lower_height, wall.height_m, vertical_steps + 1),
            indexing="ij",
        )
        xy = start[None, None, :] + along[:, :, None] * delta[None, None, :]
        chunks.append(np.column_stack((xy.reshape(-1, 2), heights.ravel())))
    return np.vstack(chunks) if chunks else np.empty((0, 3), dtype=float)


def _point_to_wall_surface_distances(points: np.ndarray, world: World3D) -> np.ndarray:
    """Exact nearest distance from each point to any static vertical rectangle."""

    nearest = np.full(len(points), np.inf, dtype=float)
    xy = points[:, :2]
    z = points[:, 2]
    for wall in world.segments:
        start = np.array([wall.x1, wall.y1], dtype=float)
        delta = np.array([wall.x2 - wall.x1, wall.y2 - wall.y1], dtype=float)
        length_sq = float(delta @ delta)
        fraction = np.clip((xy - start) @ delta / length_sq, 0.0, 1.0) if length_sq else 0.0
        closest_xy = start + np.asarray(fraction)[:, None] * delta if length_sq else start
        horizontal = np.linalg.norm(xy - closest_xy, axis=1)
        vertical = np.maximum(np.maximum(-z, z - wall.height_m), 0.0)
        nearest = np.minimum(nearest, np.hypot(horizontal, vertical))
    return nearest


def wall_point_map_metrics_3d(
    points: np.ndarray,
    world: World3D,
    spacing_m: float = WALL_SURFACE_SAMPLE_SPACING_M,
    tolerance_m: float = WALL_3D_MATCH_TOLERANCE_M,
    ground_exclusion_height_m: float = GROUND_EXCLUSION_HEIGHT_M,
) -> dict[str, float]:
    """Assess static-wall accuracy and coverage, excluding ground-like returns.

    The precision distance is exact point-to-rectangle. Coverage is sampled
    over all static wall surfaces, so unobserved walls reduce recall. Dynamic
    object returns are not part of the reference and can reduce precision.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")
    if tolerance_m <= 0.0 or not math.isfinite(tolerance_m):
        raise ValueError("3D match tolerance must be finite and positive")
    if ground_exclusion_height_m < 0.0 or not math.isfinite(ground_exclusion_height_m):
        raise ValueError("ground exclusion height must be finite and nonnegative")
    if not world.segments:
        raise ValueError("world must have static walls")
    tolerance_label = f"{tolerance_m:g}".replace(".", "p")
    precision_key = f"map3d_wall_precision_{tolerance_label}"
    recall_key = f"map3d_wall_recall_{tolerance_label}"
    f1_key = f"map3d_wall_f1_{tolerance_label}"
    reference = sample_wall_surface_points(world, spacing_m, ground_exclusion_height_m)
    if len(reference) == 0:
        raise ValueError("world has no wall surface above the ground exclusion height")
    finite = points[np.isfinite(points).all(axis=1)]
    wall_points = finite[finite[:, 2] > ground_exclusion_height_m]
    if len(wall_points) == 0:
        return {
            precision_key: 0.0,
            recall_key: 0.0,
            f1_key: 0.0,
        }
    estimated_to_true = _point_to_wall_surface_distances(wall_points, world)
    true_to_estimated = _directed_nearest_distances(reference, wall_points)
    precision = float(np.mean(estimated_to_true <= tolerance_m))
    recall = float(np.mean(true_to_estimated <= tolerance_m))
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = float(np.mean(estimated_to_true))
    completeness = float(np.mean(true_to_estimated))
    return {
        "map3d_wall_accuracy_mean_m": accuracy,
        "map3d_wall_completeness_mean_m": completeness,
        "map3d_wall_chamfer_mean_m": 0.5 * (accuracy + completeness),
        precision_key: precision,
        recall_key: recall,
        f1_key: f1,
    }
