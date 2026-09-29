import csv
import json

import numpy as np
import pytest

from slamcart3d.core.config import ExperimentConfig, LidarConfig, SimulationConfig
from slamcart3d.experiments.batch import export_batch, run_batch


def test_batch_exports_every_run_and_group_summaries(tmp_path) -> None:
    base = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )
    batch = run_batch(
        base,
        seeds=(1, 2),
        algorithms=("dead_reckoning", "wheel_imu_fusion"),
        worlds=("indoor_simple", "indoor_corridor"),
        workers=1,
    )
    assert len(batch.runs) == 8
    assert batch.workers == 1
    np.testing.assert_allclose(batch.runs[0].ground_truth, batch.runs[2].ground_truth)
    destination = export_batch(batch, tmp_path)
    with (destination / "runs.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (destination / "summary.csv").open(encoding="utf-8", newline="") as stream:
        summaries = list(csv.DictReader(stream))
    with (destination / "paired_effects.csv").open(encoding="utf-8", newline="") as stream:
        effects = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert len({row["experiment_id"] for row in rows}) == 8
    assert len([row for row in summaries if row["metric"] == "ate_rmse_m"]) == 4
    assert all(int(row["count"]) == 2 for row in summaries)
    ate_effects = [row for row in effects if row["metric"] == "ate_rmse_m"]
    assert len(ate_effects) == 2
    assert all(int(row["count"]) == 2 for row in ate_effects)
    assert all(row["cohen_dz"] == "" for row in effects if row["metric"] == "tracking_availability")
    expected = batch.runs[0].metrics["ate_rmse_m"] - batch.runs[2].metrics["ate_rmse_m"]
    assert float(ate_effects[0]["mean_difference_a_minus_b"]) == pytest.approx(
        0.5 * (expected + batch.runs[1].metrics["ate_rmse_m"] - batch.runs[3].metrics["ate_rmse_m"])
    )
    assert all((destination / "runs" / row["experiment_id"] / "metrics.csv").exists() for row in rows)
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_count"] == 8
    assert "Cohen dz" in metadata["paired_effect_method"]


def test_batch_rejects_repeated_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        run_batch(ExperimentConfig(), (1, 1), ("dead_reckoning",), ("indoor_simple",))


def test_parallel_workers_preserve_scientific_outputs() -> None:
    base = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )
    serial = run_batch(base, (1, 2), ("dead_reckoning",), ("indoor_simple",), workers=1)
    parallel = run_batch(base, (1, 2), ("dead_reckoning",), ("indoor_simple",), workers=2)
    assert parallel.workers == 2
    for one, two in zip(serial.runs, parallel.runs, strict=True):
        np.testing.assert_allclose(one.ground_truth, two.ground_truth)
        np.testing.assert_allclose(one.estimated, two.estimated)
        for metric in one.metrics:
            if metric != "runtime_s":
                assert one.metrics[metric] == two.metrics[metric]
