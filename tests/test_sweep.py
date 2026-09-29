"""Controlled-study design, pairing, threshold direction and exports."""

import csv
import json

import pytest

from slamcart3d.core.config import ExperimentConfig, LidarConfig, SimulationConfig
from slamcart3d.experiments.sweep import export_parameter_sweep, run_parameter_sweep


def test_paired_sweep_exports_every_run_and_summary(tmp_path) -> None:
    base = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )
    study = run_parameter_sweep(base, "lidar_noise_std_m", (0.0, 0.2), (0, 1))
    assert len(study.runs) == 4
    assert [run.config.seed for run in study.runs] == [0, 1, 0, 1]
    assert [run.config.degradation.lidar_noise_std_m for run in study.runs] == [0.0, 0.0, 0.2, 0.2]
    assert all(row["count"] == 2 for row in study.summary)
    target = export_parameter_sweep(study, tmp_path)
    with (target / "runs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert len(list((target / "runs").glob("EXP-*/metadata.json"))) == 4
    assert len(list(csv.DictReader((target / "summary.csv").open(newline="")))) == 2
    metadata = json.loads((target / "metadata.json").read_text())
    assert metadata["paired_design"].startswith("the same seed set")


def test_sweep_validates_scientific_design() -> None:
    base = ExperimentConfig()
    with pytest.raises(ValueError, match="strictly increasing"):
        run_parameter_sweep(base, "lidar_noise_std_m", (0.2, 0.0), (0, 1))
    with pytest.raises(ValueError, match="at least two unique"):
        run_parameter_sweep(base, "lidar_noise_std_m", (0.0, 0.2), (0, 0))
