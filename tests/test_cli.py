"""Exercise command wiring, parsing, and exported artifacts end to end."""

import json

from slamcart3d.cli import main
from slamcart3d.core.config import ExperimentConfig, IMUConfig, LidarConfig, SimulationConfig


def _short_config(tmp_path):
    config = ExperimentConfig(
        name="cli_smoke",
        lidar=LidarConfig(beam_count=12),
        imu=IMUConfig(enabled=True),
        simulation=SimulationConfig(duration_s=0.3, timestep_s=0.1),
    )
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    return path


def test_cli_run_and_dataset_exports(tmp_path) -> None:
    config = _short_config(tmp_path)
    result_root = tmp_path / "results"
    dataset_root = tmp_path / "datasets"
    assert main(["run", str(config), "--output", str(result_root)]) == 0
    assert main(["generate-dataset", str(config), "--output", str(dataset_root)]) == 0
    assert len(list(result_root.glob("EXP-*/metadata.json"))) == 1
    assert len(list(dataset_root.glob("DATASET-*/ground_truth.csv"))) == 1


def test_cli_batch_and_frontier_exports(tmp_path) -> None:
    config = _short_config(tmp_path)
    root = tmp_path / "results"
    assert (
        main(
            [
                "batch",
                str(config),
                "--seeds",
                "0,1",
                "--algorithms",
                "dead_reckoning,wheel_imu_fusion",
                "--worlds",
                "indoor_simple",
                "--workers",
                "1",
                "--output",
                str(root),
            ]
        )
        == 0
    )
    assert len(list(root.glob("BATCH-*/summary.csv"))) == 1
    assert (
        main(
            [
                "robustness-study",
                str(config),
                "--budget",
                "8",
                "--seeds",
                "0",
                "--output",
                str(root),
            ]
        )
        == 0
    )
    assert len(list(root.glob("ROBUSTNESS-*/robustness_map.csv"))) == 1
