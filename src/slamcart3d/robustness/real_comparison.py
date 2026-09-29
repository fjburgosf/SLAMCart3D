"""Compare sampling designs on real SLAM simulations and held-out seeds."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from slamcart3d import __version__
from slamcart3d.core.config import ExperimentConfig
from slamcart3d.core.provenance import source_sha256
from slamcart3d.metrics.direction import failure_margin
from slamcart3d.robustness.asrf import ASRF, ParameterSpace, classification_metrics, latin_hypercube
from slamcart3d.robustness.benchmark import grid_samples
from slamcart3d.robustness.calibration import probability_diagnostics
from slamcart3d.robustness.study import SimulationMargin


@dataclass(frozen=True)
class RealComparisonResult:
    """Independent-seed validation and per-method training observations."""

    base_config: ExperimentConfig
    parameter_names: tuple[str, str]
    lower: np.ndarray
    upper: np.ndarray
    metric_name: str
    failure_threshold: float
    training_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    design_seed: int
    budget: int
    grid_resolution: int
    rows: tuple[dict[str, float | int | str], ...]
    training_records: dict[str, tuple[dict[str, Any], ...]]
    validation_records: tuple[dict[str, Any], ...]
    calibration_bins: dict[str, tuple[dict[str, float | int], ...]]


def compare_real_sampling(
    base: ExperimentConfig,
    parameter_names: tuple[str, str],
    lower: np.ndarray,
    upper: np.ndarray,
    metric_name: str,
    failure_threshold: float,
    training_seeds: tuple[int, ...],
    validation_seeds: tuple[int, ...],
    budget: int,
    grid_resolution: int = 11,
    design_seed: int = 42,
) -> RealComparisonResult:
    """Evaluate Grid, Random, LHS, and ASRF against an independent-seed grid."""

    if budget < 8 or grid_resolution < 3:
        raise ValueError("budget must be at least 8 and validation grid resolution at least 3")
    if not training_seeds or not validation_seeds:
        raise ValueError("training and validation seeds must be non-empty")
    if len(set(training_seeds)) != len(training_seeds) or len(set(validation_seeds)) != len(validation_seeds):
        raise ValueError("seeds must be unique within each split")
    if set(training_seeds) & set(validation_seeds):
        raise ValueError("training and validation seeds must be disjoint")
    space = ParameterSpace(parameter_names, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))
    x_axis = np.linspace(space.lower[0], space.upper[0], grid_resolution)
    y_axis = np.linspace(space.lower[1], space.upper[1], grid_resolution)
    xx, yy = np.meshgrid(x_axis, y_axis)
    validation_points = np.column_stack((xx.ravel(), yy.ravel()))
    validation = SimulationMargin(base, parameter_names, metric_name, failure_threshold, validation_seeds)
    true_failed = validation(validation_points) >= 0.0
    observed_failure_fraction = np.asarray(
        [
            np.mean(
                [
                    failure_margin(float(value), metric_name, failure_threshold) >= 0.0
                    for value in record["seed_values"]
                ]
            )
            for record in validation.records
        ],
        dtype=float,
    )
    if len(np.unique(true_failed)) != 2:
        raise ValueError("validation grid must contain both success and failure; adjust bounds or threshold")

    normalized_designs = {
        "grid": grid_samples(budget),
        "random": np.random.default_rng(design_seed).random((budget, 2)),
        "lhs": latin_hypercube(budget, 2, np.random.default_rng(design_seed)),
    }
    rows: list[dict[str, float | int | str]] = []
    training_records: dict[str, tuple[dict[str, Any], ...]] = {}
    calibration_bins: dict[str, tuple[dict[str, float | int], ...]] = {}
    for method, normalized in normalized_designs.items():
        started = perf_counter()
        samples = space.denormalize(normalized)
        objective = SimulationMargin(base, parameter_names, metric_name, failure_threshold, training_seeds)
        margins = objective(samples)
        surrogate = ASRF(space, initial_samples=8, seed=design_seed)
        surrogate.samples = samples
        surrogate.failed = margins >= 0.0
        probabilities = surrogate.predict_failure_probability(validation_points)
        predicted = probabilities >= 0.5
        scores, bins = probability_diagnostics(probabilities, observed_failure_fraction)
        rows.append(
            {
                "method": method,
                "conditions": len(samples),
                "training_simulations": len(samples) * len(training_seeds),
                "runtime_s": perf_counter() - started,
                **classification_metrics(true_failed, predicted),
                **scores,
            }
        )
        training_records[method] = tuple(objective.records)
        calibration_bins[method] = tuple(bins)

    started = perf_counter()
    objective = SimulationMargin(base, parameter_names, metric_name, failure_threshold, training_seeds)
    surrogate = ASRF(space, initial_samples=8, seed=design_seed)
    asrf = surrogate.run(objective, budget=budget)
    probabilities = surrogate.predict_failure_probability(validation_points)
    predicted = probabilities >= 0.5
    scores, bins = probability_diagnostics(probabilities, observed_failure_fraction)
    rows.append(
        {
            "method": "asrf",
            "conditions": len(asrf.samples),
            "training_simulations": len(asrf.samples) * len(training_seeds),
            "runtime_s": perf_counter() - started,
            **classification_metrics(true_failed, predicted),
            **scores,
        }
    )
    training_records["asrf"] = tuple(objective.records)
    calibration_bins["asrf"] = tuple(bins)
    return RealComparisonResult(
        base,
        parameter_names,
        space.lower,
        space.upper,
        metric_name,
        failure_threshold,
        training_seeds,
        validation_seeds,
        design_seed,
        budget,
        grid_resolution,
        tuple(rows),
        training_records,
        tuple(validation.records),
        calibration_bins,
    )


def export_real_comparison(comparison: RealComparisonResult, output_root: str | Path = "results") -> Path:
    """Export all design/validation observations and reproducibility metadata."""

    created = datetime.now(timezone.utc).isoformat()
    digest = sha256((created + repr(comparison.parameter_names)).encode("utf-8")).hexdigest()[:12].upper()
    target = Path(output_root) / f"REAL-COMPARISON-{digest}"
    target.mkdir(parents=True, exist_ok=False)
    with (target / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "method",
                "conditions",
                "training_simulations",
                "runtime_s",
                "boundary_classification_error",
                "fnr",
                "fpr",
                "brier_score",
                "ece10",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison.rows)

    with (target / "reliability.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "method",
                "bin_index",
                "conditions",
                "mean_predicted_failure_probability",
                "observed_failure_fraction",
            ],
        )
        writer.writeheader()
        for method, rows in comparison.calibration_bins.items():
            writer.writerows({"method": method, **row} for row in rows)

    fieldnames = [*comparison.parameter_names, "median", "failed", "seed_values"]
    with (target / "training_observations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", *fieldnames])
        writer.writeheader()
        for method, records in comparison.training_records.items():
            for record in records:
                writer.writerow(
                    {
                        "method": method,
                        **{name: record[name] for name in comparison.parameter_names},
                        "median": record["median"],
                        "failed": record["failed"],
                        "seed_values": json.dumps(record["seed_values"]),
                    }
                )
    with (target / "validation_observations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in comparison.validation_records:
            writer.writerow(
                {
                    **{name: record[name] for name in comparison.parameter_names},
                    "median": record["median"],
                    "failed": record["failed"],
                    "seed_values": json.dumps(record["seed_values"]),
                }
            )
    metadata = {
        "comparison_id": target.name,
        "created_utc": created,
        "software_version": __version__,
        "source_sha256": source_sha256(),
        "base_config": comparison.base_config.to_dict(),
        "parameter_names": comparison.parameter_names,
        "lower": comparison.lower.tolist(),
        "upper": comparison.upper.tolist(),
        "metric": comparison.metric_name,
        "failure_threshold": comparison.failure_threshold,
        "training_seeds": comparison.training_seeds,
        "validation_seeds": comparison.validation_seeds,
        "design_seed": comparison.design_seed,
        "budget": comparison.budget,
        "validation_grid_resolution": comparison.grid_resolution,
        "validation_simulations": len(comparison.validation_records) * len(comparison.validation_seeds),
        "validation_failures": sum(bool(record["failed"]) for record in comparison.validation_records),
        "validation_successes": sum(not record["failed"] for record in comparison.validation_records),
        "interpretation": (
            "The held-out grid is an independent-seed approximation, not an exact physical frontier; "
            "all methods share the same surrogate and failure threshold."
        ),
        "probability_diagnostics": (
            "Brier score compares each predicted failure probability with all held-out binary seed outcomes; "
            "ECE10 bins predictions into ten equal-width intervals and compares average prediction with "
            "observed failure fraction. Both are descriptive on a finite grid and seed set, not proof of calibration."
        ),
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target
