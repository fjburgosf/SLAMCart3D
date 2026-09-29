"""SE(2) alignment, ATE, and relative pose error metrics."""

from __future__ import annotations

import math

import numpy as np

from slamcart3d.core.models import wrap_angle


def align_se2(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Rigidly align an estimated ``N x 3`` trajectory to a reference."""

    if reference.shape != estimate.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference and estimate must both have shape (N, 3)")
    source_xy = estimate[:, :2]
    target_xy = reference[:, :2]
    source_centered = source_xy - source_xy.mean(axis=0)
    target_centered = target_xy - target_xy.mean(axis=0)
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    aligned = estimate.copy()
    aligned[:, :2] = source_centered @ rotation + target_xy.mean(axis=0)
    yaw_offset = math.atan2(rotation[0, 1], rotation[0, 0])
    aligned[:, 2] = wrap_angle(estimate[:, 2] + yaw_offset)
    return aligned


def absolute_trajectory_error(reference: np.ndarray, estimate: np.ndarray, align: bool = True) -> float:
    """Return translational ATE RMSE in metres."""

    compared = align_se2(reference, estimate) if align else estimate
    squared = np.sum((reference[:, :2] - compared[:, :2]) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared)))


def relative_pose_error(reference: np.ndarray, estimate: np.ndarray, delta: int = 1) -> tuple[float, float]:
    """Return translational RPE [m] and rotational RPE [rad] at index delta."""

    if reference.shape != estimate.shape or delta < 1 or delta >= len(reference):
        raise ValueError("invalid trajectory shapes or delta")
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    for start in range(len(reference) - delta):
        end = start + delta
        true_yaw = reference[start, 2]
        est_yaw = estimate[start, 2]
        true_global = reference[end, :2] - reference[start, :2]
        est_global = estimate[end, :2] - estimate[start, :2]
        true_rotation = np.array(
            [[np.cos(true_yaw), np.sin(true_yaw)], [-np.sin(true_yaw), np.cos(true_yaw)]]
        )
        est_rotation = np.array([[np.cos(est_yaw), np.sin(est_yaw)], [-np.sin(est_yaw), np.cos(est_yaw)]])
        true_relative = true_rotation @ true_global
        est_relative = est_rotation @ est_global
        translation_errors.append(float(np.linalg.norm(true_relative - est_relative)))
        true_delta_yaw = float(wrap_angle(reference[end, 2] - true_yaw))
        est_delta_yaw = float(wrap_angle(estimate[end, 2] - est_yaw))
        rotation_errors.append(float(wrap_angle(est_delta_yaw - true_delta_yaw)))
    return (
        float(np.sqrt(np.mean(np.square(translation_errors)))),
        float(np.sqrt(np.mean(np.square(rotation_errors)))),
    )


def trajectory_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Compute the MVP metric set without hiding individual quantities."""

    aligned = align_se2(reference, estimate)
    rpe_translation, rpe_rotation = relative_pose_error(reference, estimate)
    final_error = float(np.linalg.norm(reference[-1, :2] - aligned[-1, :2]))
    traveled = float(np.sum(np.linalg.norm(np.diff(reference[:, :2], axis=0), axis=1)))
    return {
        "ate_rmse_m": absolute_trajectory_error(reference, estimate),
        "rpe_translation_rmse_m": rpe_translation,
        "rpe_rotation_rmse_rad": rpe_rotation,
        "final_position_error_m": final_error,
        "drift_per_meter": final_error / traveled if traveled > 0.0 else 0.0,
        "tracking_availability": float(np.mean(np.all(np.isfinite(estimate), axis=1))),
    }
