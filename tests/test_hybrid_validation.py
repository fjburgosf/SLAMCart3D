"""Fixed, paired conditions must remain reproducible and auditable."""

import csv
import json

import pytest

from slamcart3d.core.config import ExperimentConfig, LidarConfig, SimulationConfig
from slamcart3d.experiments.hybrid_validation import (
    DEFAULT_SCENARIOS,
    export_hybrid_benchmark,
    run_hybrid_benchmark,
)


def _base() -> ExperimentConfig:
    return ExperimentConfig(
        name="hybrid_validation_test",
        lidar=LidarConfig(beam_count=16),
        simulation=SimulationConfig(duration_s=0.5, timestep_s=0.1),
    )


def test_benchmark_pairs_seeds_and_exports_all_runs(tmp_path) -> None:
    benchmark = run_hybrid_benchmark(
        _base(),
        DEFAULT_SCENARIOS[:1],
        (5, 6),
        ("dead_reckoning", "adaptive_hybrid", "hybrid_no_imu"),
    )
    assert len(benchmark.runs) == 6
    assert len(benchmark.summaries) == 12
    assert len(benchmark.paired) == 8
    target = export_hybrid_benchmark(benchmark, tmp_path, plot=True)
    assert len(list((target / "runs").glob("EXP-*"))) == 6
    assert (target / "figures" / "ate_comparison.png").is_file()
    with (target / "runs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert {int(row["seed"]) for row in rows} == {5, 6}
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["design"]["seeds"] == [5, 6]


def test_benchmark_rejects_unpaired_or_duplicate_design() -> None:
    with pytest.raises(ValueError, match="two unique"):
        run_hybrid_benchmark(_base(), seeds=(5, 5))
    with pytest.raises(ValueError, match="uniquely named"):
        run_hybrid_benchmark(_base(), scenarios=(DEFAULT_SCENARIOS[0], DEFAULT_SCENARIOS[0]))
