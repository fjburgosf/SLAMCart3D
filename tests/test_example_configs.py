"""Exercise every shipped example through the public experiment path."""

from pathlib import Path

import numpy as np
import pytest

from slamcart3d.experiments.experiment import Experiment

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
EXAMPLE_CONFIGS = tuple(sorted(CONFIG_DIR.glob("example_*.yaml")))


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda path: path.stem)
def test_shipped_example_runs_with_finite_core_outputs(config_path: Path) -> None:
    experiment = Experiment.from_file(config_path)
    result = experiment.run()

    assert result.config == experiment.config
    assert result.ground_truth.shape == result.estimated.shape
    assert result.ground_truth.shape == (len(result.timestamps), 3)
    assert np.isfinite(result.ground_truth).all()
    assert np.isfinite(result.estimated).all()
    assert np.isfinite(result.metrics["ate_rmse_m"])
    assert np.isfinite(result.metrics["rpe_translation_rmse_m"])
    assert 0.0 <= result.metrics["tracking_availability"] <= 1.0
    assert result.source_sha256 and len(result.source_sha256) == 64
    assert result.metrics["ground_truth_static_wall_contact_steps"] == 0.0
    if result.config.lidar3d.enabled:
        assert result.map_points_3d.ndim == 2
        assert result.map_points_3d.shape[1] == 3
    if result.config.occupancy_grid.enabled:
        assert result.occupancy_grid is not None
        assert 0.0 <= result.metrics["occupancy_occupied_iou"] <= 1.0
