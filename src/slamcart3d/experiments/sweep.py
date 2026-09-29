"""Controlled one-factor studies with repeated seeds and auditable runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from slamcart3d import __version__
from slamcart3d.core.config import ExperimentConfig
from slamcart3d.core.provenance import source_sha256
from slamcart3d.experiments.statistics import summarize
from slamcart3d.io.export import export_result
from slamcart3d.metrics.direction import failure_margin, success_relation
from slamcart3d.simulation.runner import RunResult, run_simulation

SWEEP_PARAMETERS = (
    "lidar_noise_std_m",
    "lidar_dropout_probability",
    "wheel_rate_noise_std_rad_s",
    "imu_gyro_bias_rad_s",
    "simulation.linear_velocity_m_s",
    "simulation.yaw_rate_rad_s",
)


@dataclass(frozen=True)
class SweepResult:
    """A paired-seed design; only one condition varies between levels."""

    base_config: ExperimentConfig
    parameter_name: str
    levels: tuple[float, ...]
    seeds: tuple[int, ...]
    metric_name: str
    failure_threshold: float
    runs: tuple[RunResult, ...]
    summary: tuple[dict[str, float | int], ...]


def run_parameter_sweep(
    base: ExperimentConfig,
    parameter_name: str,
    levels: tuple[float, ...],
    seeds: tuple[int, ...],
    metric_name: str = "ate_rmse_m",
    failure_threshold: float = 0.5,
) -> SweepResult:
    """Run the same seeds at every level and summarize the chosen metric."""

    if parameter_name not in SWEEP_PARAMETERS:
        raise ValueError(f"unsupported sweep parameter: {parameter_name}")
    success_relation(metric_name)
    if len(levels) < 2 or not all(math.isfinite(value) for value in levels):
        raise ValueError("at least two finite sweep levels are required")
    if tuple(sorted(set(levels))) != levels:
        raise ValueError("sweep levels must be unique and strictly increasing")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("at least two unique nonnegative seeds are required")
    if not math.isfinite(failure_threshold):
        raise ValueError("failure threshold must be finite")
    if base.simulation.motion_profile == "scripted" and parameter_name.startswith("simulation."):
        raise ValueError("global speed and yaw axes do not modify scripted motion segments")

    runs: list[RunResult] = []
    summary: list[dict[str, float | int]] = []
    for level in levels:
        values: list[float] = []
        for seed in seeds:
            if parameter_name.startswith("simulation."):
                simulation = replace(base.simulation, **{parameter_name.removeprefix("simulation."): level})
                config = replace(base, seed=seed, simulation=simulation)
            else:
                degradation = replace(base.degradation, **{parameter_name: level})
                config = replace(base, seed=seed, degradation=degradation)
            run = run_simulation(config)
            if metric_name not in run.metrics:
                raise ValueError(f"metric unavailable in this configuration: {metric_name}")
            runs.append(run)
            values.append(float(run.metrics[metric_name]))
        statistics = summarize(values)
        failed = sum(failure_margin(value, metric_name, failure_threshold) >= 0.0 for value in values)
        summary.append(
            {
                "level": level,
                **statistics,
                "failed_runs": failed,
                "failure_fraction": failed / len(seeds),
            }
        )
    return SweepResult(
        base, parameter_name, levels, seeds, metric_name, failure_threshold, tuple(runs), tuple(summary)
    )


def export_parameter_sweep(study: SweepResult, output_root: str | Path = "results") -> Path:
    """Save the design, per-seed outcomes, summary and every underlying run."""

    created = datetime.now(timezone.utc).isoformat()
    identity = json.dumps(
        {
            "created_utc": created,
            "base_config": study.base_config.to_dict(),
            "parameter": study.parameter_name,
            "levels": study.levels,
            "seeds": study.seeds,
            "metric": study.metric_name,
            "threshold": study.failure_threshold,
        },
        sort_keys=True,
    )
    study_id = "SWEEP-" + sha256(identity.encode("utf-8")).hexdigest()[:12].upper()
    target = Path(output_root) / study_id
    target.mkdir(parents=True, exist_ok=False)
    (target / "runs").mkdir()
    metadata = {
        "study_id": study_id,
        "created_utc": created,
        "software_version": __version__,
        "source_sha256": source_sha256(),
        "base_config": study.base_config.to_dict(),
        "parameter_name": study.parameter_name,
        "levels": study.levels,
        "seeds": study.seeds,
        "metric_name": study.metric_name,
        "failure_threshold": study.failure_threshold,
        "success_relation": success_relation(study.metric_name),
        "ci95_method": "normal approximation; descriptive for small samples",
        "paired_design": "the same seed set is reused at every parameter level",
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (target / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("level", "seed", "experiment_id", study.metric_name, "failed"))
        for index, run in enumerate(study.runs):
            level = study.levels[index // len(study.seeds)]
            value = float(run.metrics[study.metric_name])
            writer.writerow(
                (
                    level,
                    run.config.seed,
                    run.experiment_id,
                    value,
                    int(failure_margin(value, study.metric_name, study.failure_threshold) >= 0.0),
                )
            )
            export_result(run, target / "runs")
    with (target / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(study.summary[0]))
        writer.writeheader()
        writer.writerows(study.summary)
    return target
