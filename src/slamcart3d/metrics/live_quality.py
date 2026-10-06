"""Explicit, bounded-cost display evaluation; never supplied to the estimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slamcart3d.metrics.map_quality import sample_wall_points
from slamcart3d.metrics.map_quality_3d import _point_to_wall_surface_distances, sample_wall_surface_points


@dataclass(frozen=True)
class LiveQuality:
    position_m: float
    heading_rad: float
    wall_distance_m: float | None
    wall_coverage: float | None
    dimension: int
    point_count: int
    sampled: bool


def evaluate_live(result, tolerance_m: float = 0.25) -> LiveQuality:
    """Use at most 2000 returns and 1500 reference samples for the live display.

    Coverage is an approximation if subsampling occurs; final exported metrics
    remain the authoritative full-data wall assessment. Ground is excluded
    from the 3D wall comparison. An empty cloud is unevaluable, not perfect.
    """

    delta = result.estimated[-1] - result.ground_truth[-1]
    position = float(np.linalg.norm(delta[:2]))
    heading = float(abs((delta[2] + np.pi) % (2 * np.pi) - np.pi))
    dimension = 3 if len(result.map_points_3d) else 2
    points = result.map_points_3d if dimension == 3 else result.map_points
    if dimension == 3:
        points = points[points[:, 2] > 0.1]
        reference = sample_wall_surface_points(result.world)
    else:
        reference = sample_wall_points(result.world)
    count = len(points)
    metrics = getattr(result, "metrics", {})
    prefix = "map3d_wall" if dimension == 3 else "map"
    if count and len(reference) and f"{prefix}_accuracy_mean_m" in metrics:
        return LiveQuality(
            position,
            heading,
            metrics[f"{prefix}_accuracy_mean_m"],
            metrics[f"{prefix}_recall_0p25"],
            dimension,
            count,
            False,
        )
    sampled = count > 2000 or len(reference) > 1500
    points = points[:: max(1, int(np.ceil(count / 2000)))]
    reference = reference[:: max(1, int(np.ceil(len(reference) / 1500)))]
    if not len(points) or not len(reference):
        return LiveQuality(position, heading, None, None, dimension, count, sampled)
    if dimension == 3:
        accuracy = float(np.mean(_point_to_wall_surface_distances(points, result.world)))
    else:
        # Exact distances to finite wall segments, not to sampled endpoints.
        nearest = np.full(len(points), np.inf)
        for wall in result.world.segments:
            start = np.array([wall.x1, wall.y1])
            vector = np.array([wall.x2 - wall.x1, wall.y2 - wall.y1])
            fraction = np.clip((points - start) @ vector / (vector @ vector), 0, 1)
            nearest = np.minimum(nearest, np.linalg.norm(points - start - fraction[:, None] * vector, axis=1))
        accuracy = float(np.mean(nearest))
    distances = []
    for offset in range(0, len(reference), 64):
        part = reference[offset : offset + 64]
        distances.extend(np.sqrt(np.min(np.sum((part[:, None] - points[None]) ** 2, axis=2), axis=1)))
    coverage = float(np.mean(np.asarray(distances) <= tolerance_m))
    return LiveQuality(position, heading, accuracy, coverage, dimension, count, sampled)
