"""Optional SRS normalization, weight sensitivity, and run integration."""

import json
from dataclasses import replace

import numpy as np
import pytest

from slamcart3d.core.config import ExperimentConfig, LidarConfig, RobustnessScoreConfig, SimulationConfig
from slamcart3d.io.export import export_result
from slamcart3d.metrics.direction import failure_margin
from slamcart3d.metrics.robustness_score import robustness_score_metrics
from slamcart3d.robustness.study import SimulationMargin, run_frontier_study
from slamcart3d.simulation.runner import run_simulation


def test_srs_is_bounded_and_has_interpretable_half_quality_scales() -> None:
    config = RobustnessScoreConfig(enabled=True)
    perfect = {
        "ate_rmse_m": 0.0,
        "rpe_translation_rmse_m": 0.0,
        "map_f1_0p25": 1.0,
        "tracking_availability": 1.0,
    }
    assert robustness_score_metrics(perfect, config)["srs"] == 1.0
    halfway = {
        "ate_rmse_m": 0.1,
        "rpe_translation_rmse_m": 0.05,
        "map_f1_0p25": 0.75,
        "tracking_availability": 0.95,
    }
    result = robustness_score_metrics(halfway, config)
    assert result["srs"] == pytest.approx(0.5)
    assert all(
        result[f"srs_{name}_quality"] == pytest.approx(0.5)
        for name in ("ate", "rpe_translation", "map_f1", "tracking_availability")
    )
    assert failure_margin(0.4, "srs", 0.5) > 0.0


def test_srs_weight_sensitivity_matches_finite_difference() -> None:
    config = RobustnessScoreConfig(enabled=True, ate_weight=2.0, rpe_translation_weight=1.0)
    metrics = {
        "ate_rmse_m": 0.2,
        "rpe_translation_rmse_m": 0.01,
        "map_f1_0p25": 0.8,
        "tracking_availability": 1.0,
    }
    result = robustness_score_metrics(metrics, config)
    epsilon = 1e-6
    perturbed = robustness_score_metrics(metrics, replace(config, ate_weight=config.ate_weight + epsilon))
    finite_difference = (perturbed["srs"] - result["srs"]) / epsilon
    assert result["srs_weight_sensitivity_ate"] == pytest.approx(finite_difference, abs=1e-7)
    weights = (
        config.ate_weight,
        config.rpe_translation_weight,
        config.map_f1_weight,
        config.tracking_availability_weight,
    )
    derivatives = (
        result["srs_weight_sensitivity_ate"],
        result["srs_weight_sensitivity_rpe_translation"],
        result["srs_weight_sensitivity_map_f1"],
        result["srs_weight_sensitivity_tracking_availability"],
    )
    assert sum(
        weight * derivative for weight, derivative in zip(weights, derivatives, strict=True)
    ) == pytest.approx(0.0)


def test_srs_is_opt_in_and_preserves_individual_metrics(tmp_path) -> None:
    base = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )
    plain = run_simulation(base)
    enabled_config = replace(base, robustness_score=RobustnessScoreConfig(enabled=True))
    scored = run_simulation(enabled_config)
    assert "srs" not in plain.metrics
    assert 0.0 <= scored.metrics["srs"] <= 1.0
    for name, value in plain.metrics.items():
        if name != "runtime_s":
            assert scored.metrics[name] == pytest.approx(value)
    assert ExperimentConfig.from_dict(enabled_config.to_dict()) == enabled_config
    target = export_result(scored, tmp_path)
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["srs_definition"]["runtime_included"] is False


def test_srs_config_rejects_unusable_weights_and_scales() -> None:
    with pytest.raises(ValueError, match="positive sum"):
        ExperimentConfig(
            robustness_score=RobustnessScoreConfig(
                ate_weight=0.0,
                rpe_translation_weight=0.0,
                map_f1_weight=0.0,
                tracking_availability_weight=0.0,
            )
        )
    with pytest.raises(ValueError, match="half-quality"):
        ExperimentConfig(robustness_score=RobustnessScoreConfig(ate_half_quality_m=0.0))
    with pytest.raises(ValueError, match="quality floors"):
        ExperimentConfig(robustness_score=RobustnessScoreConfig(map_f1_floor=1.0))


def test_asrf_srs_requires_enabled_score_and_uses_higher_is_better() -> None:
    base = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )
    with pytest.raises(ValueError, match="requires robustness_score.enabled"):
        SimulationMargin(base, ("lidar_noise_std_m", "lidar_dropout_probability"), "srs", 0.6, (0,))
    scored = replace(base, robustness_score=RobustnessScoreConfig(enabled=True))
    study = run_frontier_study(
        scored,
        ("lidar_noise_std_m", "lidar_dropout_probability"),
        np.array([0.0, 0.0]),
        np.array([0.1, 0.5]),
        "srs",
        0.6,
        (0,),
        budget=8,
        grid_resolution=5,
    )
    assert len(study.records) == 8
    assert all(record["failed"] == (record["median"] <= 0.6) for record in study.records)
