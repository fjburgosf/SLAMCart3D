"""Deterministic multi-seed, multi-algorithm, multi-world experiment batches."""

from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np

from slamcart3d import __version__
from slamcart3d.core.config import ExperimentConfig
from slamcart3d.core.provenance import source_sha256
from slamcart3d.experiments.statistics import summarize
from slamcart3d.io.export import export_result
from slamcart3d.simulation.runner import RunResult, run_simulation
from slamcart3d.worlds.geometry import make_world


@dataclass(frozen=True)
class BatchResult:
    """Ordered results plus the exact batch design that produced them."""

    base_config: ExperimentConfig
    seeds: tuple[int, ...]
    algorithms: tuple[str, ...]
    worlds: tuple[str, ...]
    workers: int
    runs: tuple[RunResult, ...]


def run_batch(
    base: ExperimentConfig,
    seeds: tuple[int, ...],
    algorithms: tuple[str, ...],
    worlds: tuple[str, ...],
    workers: int | str = 1,
) -> BatchResult:
    """Execute the Cartesian design in a fixed order, optionally in processes."""

    if not seeds or not algorithms or not worlds:
        raise ValueError("seeds, algorithms, and worlds must be non-empty")
    if (
        len(set(seeds)) != len(seeds)
        or len(set(algorithms)) != len(algorithms)
        or len(set(worlds)) != len(worlds)
    ):
        raise ValueError("batch seeds, algorithms, and worlds must be unique")
    for world in worlds:
        if world == "custom":
            if not base.simulation.world_segments:
                raise ValueError("custom batch world requires wall segments in the base configuration")
        else:
            make_world(world, base.seed)
    configurations = tuple(
        replace(
            base,
            seed=seed,
            slam=replace(base.slam, algorithm=algorithm),
            simulation=replace(base.simulation, world=world),
        )
        for world in worlds
        for algorithm in algorithms
        for seed in seeds
    )
    if workers == "auto":
        worker_count = min(os.cpu_count() or 1, len(configurations))
    elif isinstance(workers, int) and workers >= 1:
        worker_count = min(workers, len(configurations))
    else:
        raise ValueError("workers must be a positive integer or 'auto'")
    if worker_count == 1:
        runs = tuple(run_simulation(config) for config in configurations)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            runs = tuple(executor.map(run_simulation, configurations))
    return BatchResult(base, seeds, algorithms, worlds, worker_count, runs)


def _summary_rows(batch: BatchResult) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for world in batch.worlds:
        for algorithm in batch.algorithms:
            group = [
                run
                for run in batch.runs
                if run.config.simulation.world == world and run.config.slam.algorithm == algorithm
            ]
            metric_names = sorted({name for run in group for name in run.metrics})
            for metric in metric_names:
                values = [run.metrics[metric] for run in group if metric in run.metrics]
                rows.append({"world": world, "algorithm": algorithm, "metric": metric, **summarize(values)})
    return rows


def _paired_effect_rows(batch: BatchResult) -> list[dict[str, float | int | str]]:
    """Compare algorithms within each world using identical seeds."""

    runs = {
        (run.config.simulation.world, run.config.slam.algorithm, run.config.seed): run for run in batch.runs
    }
    rows: list[dict[str, float | int | str]] = []
    for world in batch.worlds:
        for algorithm_a, algorithm_b in combinations(batch.algorithms, 2):
            paired = [
                (runs[(world, algorithm_a, seed)], runs[(world, algorithm_b, seed)]) for seed in batch.seeds
            ]
            common_metrics = sorted(
                set.intersection(*(set(left.metrics) & set(right.metrics) for left, right in paired))
            )
            for metric in common_metrics:
                differences = np.asarray(
                    [left.metrics[metric] - right.metrics[metric] for left, right in paired],
                    dtype=float,
                )
                standard_deviation = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
                rows.append(
                    {
                        "world": world,
                        "algorithm_a": algorithm_a,
                        "algorithm_b": algorithm_b,
                        "metric": metric,
                        "count": len(differences),
                        "mean_difference_a_minus_b": float(np.mean(differences)),
                        "median_difference_a_minus_b": float(np.median(differences)),
                        "standard_deviation_difference": standard_deviation,
                        "cohen_dz": float(np.mean(differences)) / standard_deviation
                        if standard_deviation > 0.0
                        else "",
                    }
                )
    return rows


def export_batch(batch: BatchResult, output_root: str | Path = "results") -> Path:
    """Export every individual run and per-world/algorithm metric summaries."""

    created = datetime.now(timezone.utc).isoformat()
    identity = json.dumps(
        {
            "created_utc": created,
            "base_config": batch.base_config.to_dict(),
            "seeds": batch.seeds,
            "algorithms": batch.algorithms,
            "worlds": batch.worlds,
        },
        sort_keys=True,
    )
    batch_id = "BATCH-" + sha256(identity.encode("utf-8")).hexdigest()[:12].upper()
    target = Path(output_root) / batch_id
    target.mkdir(parents=True, exist_ok=False)
    runs_root = target / "runs"
    runs_root.mkdir()
    metadata = {
        "batch_id": batch_id,
        "created_utc": created,
        "software_version": __version__,
        "source_sha256": source_sha256(),
        "base_config": batch.base_config.to_dict(),
        "seeds": batch.seeds,
        "algorithms": batch.algorithms,
        "worlds": batch.worlds,
        "workers": batch.workers,
        "run_count": len(batch.runs),
        "summary_ci95_method": "normal approximation; interpret cautiously for small samples",
        "paired_effect_method": (
            "same-seed algorithm A minus B; Cohen dz = mean paired difference / sample SD; "
            "undefined dz is blank; descriptive only with few seeds"
        ),
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    metric_names = sorted({name for run in batch.runs for name in run.metrics})
    with (target / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["experiment_id", "world", "algorithm", "seed", *metric_names]
        )
        writer.writeheader()
        for run in batch.runs:
            export_result(run, runs_root)
            writer.writerow(
                {
                    "experiment_id": run.experiment_id,
                    "world": run.config.simulation.world,
                    "algorithm": run.config.slam.algorithm,
                    "seed": run.config.seed,
                    **run.metrics,
                }
            )
    summary = _summary_rows(batch)
    with (target / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "world",
                "algorithm",
                "metric",
                "count",
                "mean",
                "median",
                "standard_deviation",
                "iqr",
                "ci95_low",
                "ci95_high",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    with (target / "paired_effects.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "world",
                "algorithm_a",
                "algorithm_b",
                "metric",
                "count",
                "mean_difference_a_minus_b",
                "median_difference_a_minus_b",
                "standard_deviation_difference",
                "cohen_dz",
            ],
        )
        writer.writeheader()
        writer.writerows(_paired_effect_rows(batch))
    return target
