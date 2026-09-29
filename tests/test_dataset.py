import csv
import json

import numpy as np

from slamcart3d.core.config import DegradationConfig, ExperimentConfig, LidarConfig, SimulationConfig
from slamcart3d.datasets.generator import generate_dataset
from slamcart3d.experiments.experiment import Experiment
from slamcart3d.simulation.observations import ObservationStream


def test_dataset_replays_the_same_observations_as_an_experiment(tmp_path) -> None:
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        degradation=DegradationConfig(lidar_dropout_probability=0.5),
        simulation=SimulationConfig(duration_s=0.4, timestep_s=0.1),
    )
    stream = ObservationStream(config)
    first = list(stream.frames())
    second = list(stream.frames())
    for one, two in zip(first, second, strict=True):
        np.testing.assert_allclose(one.measured_lidar.ranges, two.measured_lidar.ranges, equal_nan=True)
    destination = generate_dataset(config, tmp_path)
    with (destination / "ground_truth.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    exported_truth = np.array([[float(row[key]) for key in ("x_m", "y_m", "yaw_rad")] for row in rows])
    np.testing.assert_allclose(exported_truth, Experiment(config).run().ground_truth)
    with (destination / "lidar_ranges.csv").open(encoding="utf-8", newline="") as source:
        lidar_rows = list(csv.reader(source))
    assert len(lidar_rows) == 5
    assert any(value == "" for row in lidar_rows[1:] for value in row[1:])
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["algorithm_executed"] is False
    assert metadata["seed"] == config.seed
    assert len(metadata["source_sha256"]) == 64
