"""Explicit success/failure direction for supported scalar study metrics."""

from __future__ import annotations

HIGHER_IS_BETTER = frozenset(
    {
        "tracking_availability",
        "icp_registration_availability",
        "map_precision_0p25",
        "map_recall_0p25",
        "map_f1_0p25",
        "map3d_wall_precision_0p25",
        "map3d_wall_recall_0p25",
        "map3d_wall_f1_0p25",
        "occupancy_occupied_precision",
        "occupancy_occupied_recall",
        "occupancy_occupied_f1",
        "occupancy_occupied_iou",
        "occupancy_observed_free_accuracy",
        "srs",
    }
)

LOWER_IS_BETTER = frozenset(
    {
        "ate_rmse_m",
        "position_rmse_unaligned_m",
        "position_final_unaligned_m",
        "rpe_translation_rmse_m",
        "rpe_rotation_rmse_rad",
        "final_position_error_m",
        "drift_per_meter",
        "runtime_s",
        "map_accuracy_mean_m",
        "map_completeness_mean_m",
        "map_chamfer_mean_m",
        "map3d_wall_accuracy_mean_m",
        "map3d_wall_completeness_mean_m",
        "map3d_wall_chamfer_mean_m",
    }
)


def failure_margin(value: float, metric_name: str, threshold: float) -> float:
    """Return a nonnegative margin exactly when the metric reaches failure."""

    if metric_name in HIGHER_IS_BETTER:
        return threshold - value
    if metric_name in LOWER_IS_BETTER:
        return value - threshold
    raise ValueError(f"unsupported frontier metric: {metric_name}")


def success_relation(metric_name: str) -> str:
    """Return the threshold comparison that means acceptable operation."""

    if metric_name in HIGHER_IS_BETTER:
        return ">"
    if metric_name in LOWER_IS_BETTER:
        return "<"
    raise ValueError(f"unsupported frontier metric: {metric_name}")
