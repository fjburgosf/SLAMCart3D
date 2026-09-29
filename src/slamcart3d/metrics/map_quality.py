"""Objective 2D point-map quality against known static wall geometry."""

from __future__ import annotations

import math

import numpy as np

from slamcart3d.worlds.geometry import World3D

WALL_SAMPLE_SPACING_M = 0.1
MAP_MATCH_TOLERANCE_M = 0.25


def sample_wall_points(world: World3D, spacing_m: float = WALL_SAMPLE_SPACING_M) -> np.ndarray:
    """Sample every ground-plane wall segment at approximately fixed spacing."""

    if spacing_m <= 0.0 or not math.isfinite(spacing_m):
        raise ValueError("spacing_m must be finite and positive")
    chunks: list[np.ndarray] = []
    for wall in world.segments:
        start = np.array([wall.x1, wall.y1], dtype=float)
        end = np.array([wall.x2, wall.y2], dtype=float)
        intervals = max(1, math.ceil(float(np.linalg.norm(end - start)) / spacing_m))
        t = np.linspace(0.0, 1.0, intervals + 1)
        chunks.append(start[None, :] + t[:, None] * (end - start)[None, :])
    return np.vstack(chunks) if chunks else np.empty((0, 2), dtype=float)


def _directed_nearest_distances(source: np.ndarray, target: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """Compute source-to-target Euclidean nearest distances without huge arrays."""

    target_norm = np.sum(np.square(target), axis=1)
    nearest = np.empty(len(source), dtype=float)
    for start in range(0, len(source), chunk_size):
        chunk = source[start : start + chunk_size]
        squared = np.sum(np.square(chunk), axis=1)[:, None] + target_norm[None, :] - 2.0 * chunk @ target.T
        nearest[start : start + len(chunk)] = np.sqrt(np.maximum(0.0, np.min(squared, axis=1)))
    return nearest


def point_map_metrics(
    points: np.ndarray,
    world: World3D,
    spacing_m: float = WALL_SAMPLE_SPACING_M,
    tolerance_m: float = MAP_MATCH_TOLERANCE_M,
) -> dict[str, float]:
    """Measure point accuracy and wall coverage in the unaligned world frame.

    Precision is the fraction of estimated points within ``tolerance_m`` of
    a true wall. Recall is the fraction of sampled true wall points with an
    estimated point nearby. Chamfer is the mean of the two directed mean
    nearest-neighbor distances. Unobserved/occluded walls lower recall.
    """

    if tolerance_m <= 0.0 or not math.isfinite(tolerance_m):
        raise ValueError("tolerance_m must be finite and positive")
    tolerance_label = f"{tolerance_m:g}".replace(".", "p")
    precision_key = f"map_precision_{tolerance_label}"
    recall_key = f"map_recall_{tolerance_label}"
    f1_key = f"map_f1_{tolerance_label}"
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an N x 2 array")
    reference = sample_wall_points(world, spacing_m)
    if len(reference) == 0:
        raise ValueError("world must have at least one wall segment")
    finite_points = points[np.isfinite(points).all(axis=1)]
    if len(finite_points) == 0:
        return {
            precision_key: 0.0,
            recall_key: 0.0,
            f1_key: 0.0,
        }
    estimated_to_true = _directed_nearest_distances(finite_points, reference)
    true_to_estimated = _directed_nearest_distances(reference, finite_points)
    precision = float(np.mean(estimated_to_true <= tolerance_m))
    recall = float(np.mean(true_to_estimated <= tolerance_m))
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy_m = float(np.mean(estimated_to_true))
    completeness_m = float(np.mean(true_to_estimated))
    return {
        "map_accuracy_mean_m": accuracy_m,
        "map_completeness_mean_m": completeness_m,
        "map_chamfer_mean_m": 0.5 * (accuracy_m + completeness_m),
        precision_key: precision,
        recall_key: recall,
        f1_key: f1,
    }
