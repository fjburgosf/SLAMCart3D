"""Paired, multi-condition evaluation of the proposed local SLAM method."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from slamcart3d import __version__
from slamcart3d.core.config import ExperimentConfig
from slamcart3d.core.provenance import source_sha256
from slamcart3d.experiments.statistics import summarize
from slamcart3d.io.export import export_result
from slamcart3d.simulation.runner import RunResult, run_simulation

METHODS = (
    "dead_reckoning",
    "wheel_imu_fusion",
    "icp_odometry",
    "adaptive_hybrid",
    "hybrid_no_imu",
    "hybrid_fixed_trust",
    "hybrid_no_map_gate",
)
METRICS = ("ate_rmse_m", "rpe_translation_rmse_m", "map_f1_0p25", "runtime_s")


@dataclass(frozen=True)
class BenchmarkScenario:
    """One predetermined environment and degradation condition."""

    name: str
    world: str
    lidar_noise_std_m: float
    lidar_dropout_probability: float
    wheel_rate_noise_std_rad_s: float
    imu_gyro_bias_rad_s: float = 0.0


DEFAULT_SCENARIOS = (
    BenchmarkScenario("moderate", "indoor_simple", 0.05, 0.05, 0.03),
    BenchmarkScenario("wheel_noise", "indoor_simple", 0.02, 0.0, 0.5),
    BenchmarkScenario("lidar_degradation", "indoor_corridor", 0.25, 0.4, 0.03),
    BenchmarkScenario("imu_bias", "indoor_simple", 0.02, 0.0, 0.1, 0.08),
)


@dataclass(frozen=True)
class HybridBenchmark:
    """All paired runs and descriptive summaries; no superiority claim is implied."""

    base: ExperimentConfig
    scenarios: tuple[BenchmarkScenario, ...]
    seeds: tuple[int, ...]
    methods: tuple[str, ...]
    runs: tuple[tuple[str, str, RunResult], ...]
    summaries: tuple[dict[str, float | int | str], ...]
    paired: tuple[dict[str, float | int | str], ...]


def _method_config(config: ExperimentConfig, method: str) -> ExperimentConfig:
    if method not in METHODS:
        raise ValueError(f"unsupported benchmark method: {method}")
    if method.startswith("hybrid_"):
        settings = {
            "hybrid_use_imu": method != "hybrid_no_imu",
            "hybrid_adaptive_trust": method != "hybrid_fixed_trust",
            "hybrid_gate_map": method != "hybrid_no_map_gate",
        }
        return replace(config, slam=replace(config.slam, algorithm="adaptive_hybrid", **settings))
    return replace(config, slam=replace(config.slam, algorithm=method))


def run_hybrid_benchmark(
    base: ExperimentConfig,
    scenarios: tuple[BenchmarkScenario, ...] = DEFAULT_SCENARIOS,
    seeds: tuple[int, ...] = (5, 6, 7, 8, 9),
    methods: tuple[str, ...] = METHODS,
) -> HybridBenchmark:
    """Run each method with identical seed and condition in a fixed order."""

    if not scenarios or len(set(s.name for s in scenarios)) != len(scenarios):
        raise ValueError("scenarios must be non-empty and uniquely named")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("at least two unique nonnegative seeds are required")
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError("methods must be non-empty, unique and supported")
    runs: list[tuple[str, str, RunResult]] = []
    summaries: list[dict[str, float | int | str]] = []
    paired: list[dict[str, float | int | str]] = []
    for scenario in scenarios:
        condition = replace(
            base,
            simulation=replace(base.simulation, world=scenario.world),
            degradation=replace(
                base.degradation,
                lidar_noise_std_m=scenario.lidar_noise_std_m,
                lidar_dropout_probability=scenario.lidar_dropout_probability,
                wheel_rate_noise_std_rad_s=scenario.wheel_rate_noise_std_rad_s,
                imu_gyro_bias_rad_s=scenario.imu_gyro_bias_rad_s,
            ),
        )
        by_method: dict[str, list[RunResult]] = {}
        for method in methods:
            method_runs = [
                run_simulation(_method_config(replace(condition, seed=seed), method)) for seed in seeds
            ]
            by_method[method] = method_runs
            runs.extend((scenario.name, method, run) for run in method_runs)
            for metric in METRICS:
                summaries.append(
                    {
                        "scenario": scenario.name,
                        "method": method,
                        "metric": metric,
                        **summarize([run.metrics[metric] for run in method_runs]),
                    }
                )
        if "adaptive_hybrid" in by_method:
            for method in methods:
                if method == "adaptive_hybrid":
                    continue
                for metric in METRICS:
                    differences = [
                        proposed.metrics[metric] - reference.metrics[metric]
                        for proposed, reference in zip(
                            by_method["adaptive_hybrid"], by_method[method], strict=True
                        )
                    ]
                    paired.append(
                        {
                            "scenario": scenario.name,
                            "reference_method": method,
                            "metric": metric,
                            **summarize(differences),
                        }
                    )
    return HybridBenchmark(base, scenarios, seeds, methods, tuple(runs), tuple(summaries), tuple(paired))


def export_hybrid_benchmark(
    benchmark: HybridBenchmark, output_root: str | Path = "results", *, plot: bool = False
) -> Path:
    """Export exact designs, individual runs, summaries and paired contrasts."""

    created = datetime.now(timezone.utc).isoformat()
    design = {
        "created_utc": created,
        "base_config": benchmark.base.to_dict(),
        "scenarios": [scenario.__dict__ for scenario in benchmark.scenarios],
        "seeds": benchmark.seeds,
        "methods": benchmark.methods,
    }
    study_id = "HYBRID-" + sha256(json.dumps(design, sort_keys=True).encode("utf-8")).hexdigest()[:12].upper()
    target = Path(output_root) / study_id
    target.mkdir(parents=True, exist_ok=False)
    (target / "runs").mkdir()
    metadata = {
        "study_id": study_id,
        "software_version": __version__,
        "source_sha256": source_sha256(),
        "design": design,
        "statistical_interpretation": (
            "descriptive means and normal-approximation intervals over the recorded seeds; paired differences "
            "are proposed minus reference; no significance or general superiority is asserted"
        ),
        "method_scope": "planar local scan-to-map SLAM without global loop closure",
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (target / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scenario", "method", "seed", "experiment_id", *METRICS))
        for scenario, method, run in benchmark.runs:
            writer.writerow(
                (
                    scenario,
                    method,
                    run.config.seed,
                    run.experiment_id,
                    *(run.metrics[metric] for metric in METRICS),
                )
            )
            export_result(run, target / "runs")
    for filename, rows in (("summary.csv", benchmark.summaries), ("paired_effects.csv", benchmark.paired)):
        with (target / filename).open("w", encoding="utf-8", newline="") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    if plot:
        from slamcart3d.visualization.hybrid_study import save_hybrid_ate_figure

        (target / "figures").mkdir()
        save_hybrid_ate_figure(benchmark, target / "figures" / "ate_comparison.png")
    return target
