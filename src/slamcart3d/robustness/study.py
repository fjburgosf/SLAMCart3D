"""Connect ASRF to real, repeated SLAMCart3D simulations."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from slamcart3d import __version__
from slamcart3d.core.config import DegradationConfig, ExperimentConfig
from slamcart3d.core.provenance import source_sha256
from slamcart3d.experiments.statistics import summarize
from slamcart3d.metrics.direction import failure_margin, success_relation
from slamcart3d.robustness.asrf import ASRF, ASRFResult, ParameterSpace
from slamcart3d.robustness.sensitivity import sampled_sensitivity
from slamcart3d.simulation.runner import run_simulation

SIMULATION_PARAMETER_NAMES = (
    "simulation.linear_velocity_m_s",
    "simulation.yaw_rate_rad_s",
    "simulation.motion_frequency_hz",
)
STUDY_DEGRADATION_PARAMETER_NAMES = tuple(
    item.name
    for item in fields(DegradationConfig)
    if not item.name.startswith("camera_")
    and item.name
    not in {
        "dropout_time_profile",
        "dropout_region_enabled",
        "dropout_region_x_min_m",
        "dropout_region_x_max_m",
        "dropout_region_y_min_m",
        "dropout_region_y_max_m",
    }
)


@dataclass(frozen=True)
class FrontierStudyResult:
    """ASRF observations, probability grid, and repeated-run statistics."""

    asrf: ASRFResult
    records: tuple[dict[str, Any], ...]
    parameter_names: tuple[str, str]
    x_axis: np.ndarray
    y_axis: np.ndarray
    failure_probability: np.ndarray
    failure_uncertainty: np.ndarray
    metric_name: str
    failure_threshold: float
    seeds: tuple[int, ...]
    base_config: ExperimentConfig
    asrf_seed: int
    asrf_parameters: dict[str, float | int]

    @property
    def success_probability(self) -> np.ndarray:
        """Complement of the exploratory kernel-smoothed failure estimate."""

        return 1.0 - self.failure_probability


class SimulationMargin:
    """Callable performance margin evaluated over multiple deterministic seeds."""

    def __init__(
        self,
        base: ExperimentConfig,
        parameter_names: tuple[str, str],
        metric_name: str,
        failure_threshold: float,
        seeds: tuple[int, ...],
    ) -> None:
        if len(seeds) < 1:
            raise ValueError("at least one seed is required")
        if len(set(parameter_names)) != 2:
            raise ValueError("choose two distinct degradation parameters")
        unknown = [
            name
            for name in parameter_names
            if name not in STUDY_DEGRADATION_PARAMETER_NAMES and name not in SIMULATION_PARAMETER_NAMES
        ]
        if unknown:
            raise ValueError(f"unknown study parameters: {unknown}")
        if (
            "simulation.motion_frequency_hz" in parameter_names
            and base.simulation.motion_profile != "sinusoidal_yaw"
        ):
            raise ValueError("motion_frequency_hz requires the sinusoidal_yaw motion profile")
        if base.simulation.motion_profile == "scripted" and any(
            name in {"simulation.linear_velocity_m_s", "simulation.yaw_rate_rad_s"}
            for name in parameter_names
        ):
            raise ValueError("global speed and yaw-rate axes do not modify scripted motion segments")
        success_relation(metric_name)
        if metric_name == "srs" and not base.robustness_score.enabled:
            raise ValueError("SRS metric requires robustness_score.enabled=true")
        if metric_name.startswith("occupancy_") and not base.occupancy_grid.enabled:
            raise ValueError("occupancy metric requires occupancy_grid.enabled=true")
        self.base = base
        self.parameter_names = parameter_names
        self.metric_name = metric_name
        self.failure_threshold = failure_threshold
        self.seeds = seeds
        self.records: list[dict[str, Any]] = []

    def __call__(self, points: np.ndarray) -> np.ndarray:
        margins: list[float] = []
        for point in np.atleast_2d(points):
            parameter_values = dict(zip(self.parameter_names, map(float, point), strict=True))
            degradation_values = {
                name: value for name, value in parameter_values.items() if not name.startswith("simulation.")
            }
            simulation_values = {
                name.removeprefix("simulation."): value
                for name, value in parameter_values.items()
                if name.startswith("simulation.")
            }
            values: list[float] = []
            for seed in self.seeds:
                degradation = replace(self.base.degradation, **degradation_values)
                simulation = replace(self.base.simulation, **simulation_values)
                config = replace(self.base, seed=seed, degradation=degradation, simulation=simulation)
                metrics = run_simulation(config).metrics
                if self.metric_name not in metrics:
                    raise ValueError(f"unknown metric: {self.metric_name}")
                values.append(float(metrics[self.metric_name]))
            statistics = summarize(values)
            margin = failure_margin(float(statistics["median"]), self.metric_name, self.failure_threshold)
            self.records.append(
                {
                    **parameter_values,
                    **statistics,
                    "metric": self.metric_name,
                    "failure_threshold": self.failure_threshold,
                    "performance_margin": margin,
                    "failed": margin >= 0.0,
                    "seed_values": values,
                }
            )
            margins.append(margin)
        return np.asarray(margins)


def run_frontier_study(
    base: ExperimentConfig,
    parameter_names: tuple[str, str],
    lower: np.ndarray,
    upper: np.ndarray,
    metric_name: str,
    failure_threshold: float,
    seeds: tuple[int, ...],
    budget: int,
    asrf_seed: int = 42,
    grid_resolution: int = 81,
) -> FrontierStudyResult:
    """Estimate a two-dimensional real-simulation robustness map."""

    space = ParameterSpace(parameter_names, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))
    objective = SimulationMargin(base, parameter_names, metric_name, failure_threshold, seeds)
    model = ASRF(space, initial_samples=min(8, budget), seed=asrf_seed)
    result = model.run(objective, budget=budget)
    x_axis = np.linspace(lower[0], upper[0], grid_resolution)
    y_axis = np.linspace(lower[1], upper[1], grid_resolution)
    xx, yy = np.meshgrid(x_axis, y_axis)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    probability = model.predict_failure_probability(grid_points).reshape(xx.shape)
    uncertainty = model.predict_failure_uncertainty(grid_points).reshape(xx.shape)
    return FrontierStudyResult(
        asrf=result,
        records=tuple(objective.records),
        parameter_names=parameter_names,
        x_axis=x_axis,
        y_axis=y_axis,
        failure_probability=probability,
        failure_uncertainty=uncertainty,
        metric_name=metric_name,
        failure_threshold=failure_threshold,
        seeds=seeds,
        base_config=base,
        asrf_seed=asrf_seed,
        asrf_parameters={
            "initial_samples": model.initial_samples,
            "bandwidth": model.bandwidth,
            "boundary_weight": model.boundary_weight,
            "coverage_weight": model.coverage_weight,
            "candidate_count": model.candidate_count,
            "convergence_tolerance": 0.01,
            "convergence_patience": 8,
        },
    )


def export_frontier_study(
    study: FrontierStudyResult, output_root: str | Path = "results", plot: bool = False
) -> Path:
    """Export observations, probability surface, metadata, and robustness figure."""

    created = datetime.now(timezone.utc).isoformat()
    digest = sha256((created + repr(study.parameter_names)).encode("utf-8")).hexdigest()[:12].upper()
    target = Path(output_root) / f"ROBUSTNESS-{digest}"
    target.mkdir(parents=True, exist_ok=False)
    configuration = {
        "experiment": study.base_config.to_dict(),
        "frontier_study": {
            "parameter_names": list(study.parameter_names),
            "lower": [float(study.x_axis[0]), float(study.y_axis[0])],
            "upper": [float(study.x_axis[-1]), float(study.y_axis[-1])],
            "metric": study.metric_name,
            "failure_threshold": study.failure_threshold,
            "success_relation": success_relation(study.metric_name),
            "seeds": list(study.seeds),
            "asrf_seed": study.asrf_seed,
            "asrf_parameters": study.asrf_parameters,
            "grid_resolution": len(study.x_axis),
        },
    }
    (target / "config.yaml").write_text(
        yaml.safe_dump(configuration, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    fieldnames = [
        *study.parameter_names,
        "count",
        "mean",
        "median",
        "standard_deviation",
        "iqr",
        "ci95_low",
        "ci95_high",
        "metric",
        "failure_threshold",
        "performance_margin",
        "failed",
        "seed_values",
    ]
    with (target / "observations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(study.records)
    with (target / "seed_runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*study.parameter_names, "seed", "metric", "value", "failure_threshold", "failed"])
        for record in study.records:
            for seed, value in zip(study.seeds, record["seed_values"], strict=True):
                writer.writerow(
                    [
                        *(record[name] for name in study.parameter_names),
                        seed,
                        study.metric_name,
                        value,
                        study.failure_threshold,
                        failure_margin(value, study.metric_name, study.failure_threshold) >= 0.0,
                    ]
                )
    with (target / "sensitivity.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["parameter", "metric", "finite_pair_count", "spearman_rho"]
        )
        writer.writeheader()
        writer.writerows(sampled_sensitivity(study.records, study.parameter_names))
    xx, yy = np.meshgrid(study.x_axis, study.y_axis)
    grid = np.column_stack(
        (xx.ravel(), yy.ravel(), study.success_probability.ravel(), study.failure_probability.ravel())
    )
    np.savetxt(
        target / "robustness_map.csv",
        grid,
        delimiter=",",
        header=f"{study.parameter_names[0]},{study.parameter_names[1]},success_probability,failure_probability",
        comments="",
    )
    uncertainty_grid = np.column_stack((xx.ravel(), yy.ravel(), study.failure_uncertainty.ravel()))
    np.savetxt(
        target / "robustness_uncertainty.csv",
        uncertainty_grid,
        delimiter=",",
        header=f"{study.parameter_names[0]},{study.parameter_names[1]},posterior_std",
        comments="",
    )
    metadata = {
        "study_id": target.name,
        "created_utc": created,
        "software_version": __version__,
        "source_sha256": source_sha256(),
        "parameters": study.parameter_names,
        "metric": study.metric_name,
        "failure_threshold": study.failure_threshold,
        "seeds": study.seeds,
        "simulations": len(study.records) * len(study.seeds),
        "adaptive_observations": len(study.records),
        "asrf_seed": study.asrf_seed,
        "sensitivity_interpretation": (
            "Spearman rank association on adaptively sampled conditions; exploratory, not causal"
        ),
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if plot:
        from slamcart3d.visualization.plots import save_robustness_map, save_uncertainty_map

        save_robustness_map(study, target / "robustness_map.png")
        save_uncertainty_map(study, target / "robustness_uncertainty.png")
    return target
