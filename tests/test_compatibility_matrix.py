"""Every built-in estimator must run in every built-in world."""

from dataclasses import replace

import numpy as np
import pytest

from slamcart3d.core.config import ExperimentConfig
from slamcart3d.simulation.runner import run_simulation


@pytest.mark.parametrize(
    "world",
    [
        "indoor_simple",
        "indoor_corridor",
        "indoor_complex",
        "warehouse_shelves",
        "indoor_maze",
        "indoor_dynamic",
        "outdoor_blocks",
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        "dead_reckoning",
        "wheel_imu_fusion",
        "icp_odometry",
        "adaptive_hybrid",
        "posegraph_hybrid",
        "hybrid_3d",
    ],
)
def test_world_algorithm_combination_has_finite_scientific_outputs(world: str, algorithm: str) -> None:
    base = ExperimentConfig()
    config = replace(
        base,
        lidar=replace(base.lidar, beam_count=16),
        simulation=replace(base.simulation, world=world, duration_s=0.3),
        slam=replace(base.slam, algorithm=algorithm),
        lidar3d=replace(base.lidar3d, enabled=algorithm == "hybrid_3d"),
    )
    result = run_simulation(config)
    assert len(result.ground_truth) == len(result.estimated) == 4
    assert np.isfinite(result.ground_truth).all()
    assert np.isfinite(result.estimated).all()
    assert np.isfinite(result.metrics["ate_rmse_m"])
    assert 0.0 <= result.metrics["map_recall_0p25"] <= 1.0
