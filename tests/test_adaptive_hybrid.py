"""Checks for the proposed hybrid estimator and its ablation switches."""

from dataclasses import replace

import numpy as np

from slamcart3d.core.config import (
    DegradationConfig,
    ExperimentConfig,
    LidarConfig,
    SimulationConfig,
    SLAMConfig,
)
from slamcart3d.core.models import Pose2D
from slamcart3d.simulation.runner import run_simulation
from slamcart3d.slam.adaptive_hybrid import _register_scan_to_map


def test_scan_to_map_registration_reduces_a_known_offset() -> None:
    angles = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    scan = np.column_stack((3.0 * np.cos(angles), 2.0 * np.sin(angles)))
    map_points = scan.copy()
    prior = Pose2D(0.12, -0.08, 0.03)
    corrected, rmse, fraction = _register_scan_to_map(scan, map_points, prior, 3.0, 0.6, 12)
    assert corrected is not None
    assert rmse < 0.08
    assert fraction > 0.9
    assert np.hypot(corrected.x, corrected.y) < np.hypot(prior.x, prior.y)


def test_hybrid_is_deterministic_and_reports_registration_quality() -> None:
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=60),
        simulation=SimulationConfig(duration_s=1.0, timestep_s=0.1),
        slam=SLAMConfig(algorithm="adaptive_hybrid"),
    )
    first = run_simulation(config)
    second = run_simulation(config)
    np.testing.assert_array_equal(first.ground_truth, second.ground_truth)
    np.testing.assert_array_equal(first.estimated, second.estimated)
    np.testing.assert_array_equal(first.map_points, second.map_points)
    assert first.diagnostics["accepted_map_registrations"] > 0
    assert first.metrics["icp_registration_availability"] > 0
    assert 0 <= first.diagnostics["online_trust"] <= 1


def test_blind_lidar_falls_back_and_ablation_flags_remain_executable() -> None:
    base = ExperimentConfig(
        lidar=LidarConfig(beam_count=12),
        degradation=DegradationConfig(lidar_dropout_probability=1.0),
        simulation=SimulationConfig(duration_s=0.4, timestep_s=0.1),
        slam=SLAMConfig(algorithm="adaptive_hybrid"),
    )
    blind = run_simulation(base)
    assert len(blind.map_points) == 0
    assert blind.diagnostics["accepted_map_registrations"] == 0
    assert np.isfinite(blind.estimated).all()
    for field in ("hybrid_use_imu", "hybrid_adaptive_trust", "hybrid_gate_map"):
        config = replace(base, degradation=DegradationConfig(), slam=replace(base.slam, **{field: False}))
        result = run_simulation(config)
        assert np.isfinite(result.estimated).all()
        assert len(result.map_points) > 0
