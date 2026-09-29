"""Optional interpretable composite score with exact local weight sensitivity."""

from __future__ import annotations

import math

from slamcart3d.core.config import RobustnessScoreConfig


def robustness_score_metrics(metrics: dict[str, float], config: RobustnessScoreConfig) -> dict[str, float]:
    """Map four individual metrics to [0, 1] and form a weighted mean.

    Error half-quality scales yield 0.5 quality at the configured error.
    Quality floors yield zero at/below the configured minimum. Sensitivities
    are exact partial derivatives with respect to raw, unnormalized weights.
    """

    required = (
        "ate_rmse_m",
        "rpe_translation_rmse_m",
        "map_f1_0p25",
        "tracking_availability",
    )
    weights_to_check = (
        config.ate_weight,
        config.rpe_translation_weight,
        config.map_f1_weight,
        config.tracking_availability_weight,
    )
    if (
        config.ate_half_quality_m <= 0.0
        or config.rpe_translation_half_quality_m <= 0.0
        or not 0.0 <= config.map_f1_floor < 1.0
        or not 0.0 <= config.tracking_availability_floor < 1.0
        or any(weight < 0.0 for weight in weights_to_check)
        or sum(weights_to_check) <= 0.0
    ):
        raise ValueError("SRS configuration scales, floors, or weights are invalid")
    values = {name: float(metrics[name]) for name in required}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("SRS requires finite component metrics")
    if values["ate_rmse_m"] < 0.0 or values["rpe_translation_rmse_m"] < 0.0:
        raise ValueError("SRS error metrics must be nonnegative")
    if not 0.0 <= values["map_f1_0p25"] <= 1.0 or not 0.0 <= values["tracking_availability"] <= 1.0:
        raise ValueError("SRS quality metrics must be in [0, 1]")
    qualities = {
        "ate": config.ate_half_quality_m / (config.ate_half_quality_m + values["ate_rmse_m"]),
        "rpe_translation": config.rpe_translation_half_quality_m
        / (config.rpe_translation_half_quality_m + values["rpe_translation_rmse_m"]),
        "map_f1": max(
            0.0,
            (values["map_f1_0p25"] - config.map_f1_floor) / (1.0 - config.map_f1_floor),
        ),
        "tracking_availability": max(
            0.0,
            (values["tracking_availability"] - config.tracking_availability_floor)
            / (1.0 - config.tracking_availability_floor),
        ),
    }
    weights = {
        "ate": config.ate_weight,
        "rpe_translation": config.rpe_translation_weight,
        "map_f1": config.map_f1_weight,
        "tracking_availability": config.tracking_availability_weight,
    }
    total_weight = sum(weights.values())
    score = sum(weights[name] * quality for name, quality in qualities.items()) / total_weight
    return {
        "srs": float(score),
        **{f"srs_{name}_quality": float(quality) for name, quality in qualities.items()},
        **{
            f"srs_weight_sensitivity_{name}": float((quality - score) / total_weight)
            for name, quality in qualities.items()
        },
    }
