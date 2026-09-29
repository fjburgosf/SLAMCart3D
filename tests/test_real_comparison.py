"""Real-simulation sampler comparison with independent validation seeds."""

import csv
import json

import numpy as np
import pytest

from slamcart3d.core.config import ExperimentConfig, LidarConfig, SimulationConfig
from slamcart3d.robustness.real_comparison import compare_real_sampling, export_real_comparison


def _short_config() -> ExperimentConfig:
    return ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )


def test_real_comparison_exports_every_design_and_validation_condition(tmp_path) -> None:
    comparison = compare_real_sampling(
        _short_config(),
        ("wheel_rate_noise_std_rad_s", "odometry_dropout_probability"),
        np.array([0.0, 0.0]),
        np.array([0.1, 1.0]),
        "ate_rmse_m",
        0.01,
        (0, 1),
        (2, 3),
        budget=8,
        grid_resolution=3,
    )
    assert {row["method"] for row in comparison.rows} == {"grid", "random", "lhs", "asrf"}
    assert all(row["conditions"] == 8 for row in comparison.rows)
    assert all(row["training_simulations"] == 16 for row in comparison.rows)
    assert all(0.0 <= row["brier_score"] <= 1.0 for row in comparison.rows)
    assert all(0.0 <= row["ece10"] <= 1.0 for row in comparison.rows)
    assert set(comparison.calibration_bins) == {"grid", "random", "lhs", "asrf"}
    target = export_real_comparison(comparison, tmp_path)
    with (target / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    with (target / "reliability.csv").open(encoding="utf-8", newline="") as stream:
        reliability = list(csv.DictReader(stream))
    with (target / "training_observations.csv").open(encoding="utf-8", newline="") as stream:
        training = list(csv.DictReader(stream))
    with (target / "validation_observations.csv").open(encoding="utf-8", newline="") as stream:
        validation = list(csv.DictReader(stream))
    assert len(training) == 32
    assert len(validation) == 9
    assert len(summary) == 4 and "brier_score" in summary[0]
    assert reliability and {row["method"] for row in reliability} == {"grid", "random", "lhs", "asrf"}
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["validation_simulations"] == 18
    assert len(metadata["source_sha256"]) == 64
    assert "Brier score" in metadata["probability_diagnostics"]


def test_real_comparison_rejects_seed_leakage() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        compare_real_sampling(
            _short_config(),
            ("wheel_rate_noise_std_rad_s", "odometry_dropout_probability"),
            np.array([0.0, 0.0]),
            np.array([0.1, 0.1]),
            "ate_rmse_m",
            0.01,
            (0, 1),
            (1, 2),
            budget=8,
            grid_resolution=3,
        )


def test_real_comparison_rejects_validation_without_both_classes() -> None:
    with pytest.raises(ValueError, match="both success and failure"):
        compare_real_sampling(
            _short_config(),
            ("wheel_rate_noise_std_rad_s", "odometry_dropout_probability"),
            np.array([0.0, 0.0]),
            np.array([0.1, 0.1]),
            "ate_rmse_m",
            10.0,
            (0,),
            (1,),
            budget=8,
            grid_resolution=3,
        )
