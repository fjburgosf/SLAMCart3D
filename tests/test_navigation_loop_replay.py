"""Behavioral checks for exploration, loop closure, replay and evaluation."""

import json
from dataclasses import replace

import numpy as np
import pytest

from slamcart3d.core.config import ExperimentConfig, NavigationConfig, WallConfig
from slamcart3d.core.models import LidarScan, Pose2D
from slamcart3d.datasets.generator import generate_dataset
from slamcart3d.datasets.replay import export_replay, load_carmen, replay_recording
from slamcart3d.io.export import export_result
from slamcart3d.metrics.live_quality import evaluate_live
from slamcart3d.metrics.trajectory import trajectory_metrics
from slamcart3d.navigation.frontier import FrontierExplorer
from slamcart3d.simulation.runner import run_simulation
from slamcart3d.slam.loop_closure import PoseEdge, graph_cost, optimize_pose_graph, relative_pose


@pytest.mark.parametrize(
    "section,changes",
    [
        ("navigation", {"resolution_m": 0.01}),
        ("navigation", {"max_yaw_rate_rad_s": 0}),
        ("navigation", {"safety_margin_m": -1}),
        ("navigation", {"replan_steps": 0}),
        ("navigation", {"replan_steps": 1.2}),
        ("evaluation", {"coverage_target": 1.1}),
        ("evaluation", {"position_limit_m": 0}),
        ("evaluation", {"wall_distance_limit_m": float("nan")}),
    ],
)
def test_new_parameters_are_validated(section, changes):
    base = ExperimentConfig()
    with pytest.raises(ValueError):
        replace(base, **{section: replace(getattr(base, section), **changes)})


def test_navigation_stops_for_missing_measurements_or_near_obstacles():
    planner = FrontierExplorer(NavigationConfig(), 0.42, 0.4, 12)
    pose = Pose2D(5, 4, 0)
    angles = np.linspace(-1, 1, 20)
    assert planner.command(pose, LidarScan(0, angles, np.full(20, np.nan))) == (0, 0)
    speed, turn = planner.command(pose, LidarScan(0, angles, np.full(20, 0.4)))
    assert speed == 0 and turn > 0 and planner.reactive_stops == 1


def test_frontier_route_is_free_and_avoids_inflated_obstacles():
    planner = FrontierExplorer(NavigationConfig(), 0.3, 0.4, 12)
    planner.grid.free = {(x, y) for x in range(-15, 16) for y in range(-15, 16)}
    planner.grid.occupied = {(3, 0)}
    planner.grid.free.discard((3, 0))
    planner._plan(Pose2D(0, 0, 0))
    assert planner.route
    assert all(cell in planner.grid.free for cell in planner.route)
    assert all(np.linalg.norm(np.array(cell) - [3, 0]) * 0.25 > planner.radius_m for cell in planner.route)
    assert all(
        sum(abs(a - b) for a, b in zip(first, second)) == 1
        for first, second in zip(planner.route[:-1], planner.route[1:])
    )


def test_autonomous_exploration_moves_reproducibly():
    base = ExperimentConfig()
    config = replace(
        base,
        lidar=replace(base.lidar, beam_count=60),
        slam=replace(base.slam, algorithm="adaptive_hybrid"),
        simulation=replace(base.simulation, motion_profile="frontier", duration_s=10),
    )
    first, second = run_simulation(config), run_simulation(config)
    assert np.array_equal(first.ground_truth, second.ground_truth)
    assert np.linalg.norm(first.ground_truth[-1, :2] - first.ground_truth[0, :2]) > 0.5
    assert first.diagnostics["navigation_replans"] > 0
    assert first.diagnostics["navigation_pose_source"] == "estimated pose; no world geometry"
    assert first.metrics["ground_truth_static_wall_contact_steps"] == 0
    assert first.occupancy_grid is not None


def test_pose_graph_cost_decreases_and_anchor_remains_fixed():
    exact = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, np.pi / 2], [0.0, 0, 0]])
    initial = exact.copy()
    initial[1:, 0] += [0.1, 0.2, 0.3]
    edges = [
        PoseEdge(i, i + 1, relative_pose(initial[i], initial[i + 1]), np.full(3, 100.0)) for i in range(3)
    ]
    edges.append(PoseEdge(0, 3, relative_pose(exact[0], exact[3]), np.full(3, 1000.0), True))
    optimized = optimize_pose_graph(initial, edges)
    assert np.array_equal(optimized[0], initial[0])
    assert graph_cost(optimized, edges) < graph_cost(initial, edges)
    assert np.linalg.norm(optimized[-1, :2]) < np.linalg.norm(initial[-1, :2])


def test_loop_run_retains_online_trace_and_rebuilds_maps(tmp_path):
    base = ExperimentConfig()
    walls = (
        WallConfig(0, 0, 12, 0),
        WallConfig(12, 0, 12, 10),
        WallConfig(12, 10, 0, 10),
        WallConfig(0, 10, 0, 0),
    )
    config = replace(
        base,
        simulation=replace(
            base.simulation, world="custom", world_segments=walls, duration_s=34, yaw_rate_rad_s=0.4
        ),
        lidar=replace(base.lidar, beam_count=60),
        slam=replace(base.slam, algorithm="posegraph_hybrid"),
        lidar3d=replace(base.lidar3d, enabled=True, azimuth_beams=6, elevation_beams=3),
    )
    result = run_simulation(config)
    assert result.online_estimated.shape == result.estimated.shape
    assert result.diagnostics["accepted_loops"] > 0
    assert result.diagnostics["posegraph_cost_after"] <= result.diagnostics["posegraph_cost_before"]
    assert np.isfinite(result.map_points_3d).all()
    target = export_result(result, tmp_path)
    assert (target / "trajectory_online.csv").is_file()
    metadata = json.loads((target / "metadata.json").read_text())
    assert metadata["trajectory_evaluation"]["final_trajectory"] == "pose graph optimized"


def test_direct_error_does_not_hide_world_frame_offset():
    reference = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    metrics = trajectory_metrics(reference, reference + [3, 0, 0])
    assert metrics["ate_rmse_m"] == pytest.approx(0)
    assert metrics["position_rmse_unaligned_m"] == pytest.approx(3)


def test_empty_live_map_is_not_claimed_perfect():
    base = ExperimentConfig()
    result = run_simulation(
        replace(
            base,
            simulation=replace(base.simulation, duration_s=0.1),
            degradation=replace(base.degradation, lidar_dropout_probability=1),
        )
    )
    quality = evaluate_live(result)
    assert quality.wall_distance_m is None and quality.wall_coverage is None


def test_measured_csv_replay_ignores_ideal_channels(tmp_path):
    base = ExperimentConfig()
    config = replace(
        base, simulation=replace(base.simulation, duration_s=0.3), lidar=replace(base.lidar, beam_count=8)
    )
    dataset = generate_dataset(config, tmp_path)
    replay = replay_recording(dataset, config, reference_path=dataset / "ground_truth.csv", max_frames=3)
    np.testing.assert_allclose(replay.estimated, run_simulation(config).estimated, atol=1e-12)
    assert replay.metadata["data_origin"] == "synthetic_simulation"
    assert replay.reference is not None and "ate_rmse_m" in replay.metrics
    assert export_replay(replay, tmp_path).is_dir()
    (dataset / "lidar_ideal_ranges.csv").write_text("not used", encoding="utf-8")
    replay2 = replay_recording(dataset, config, max_frames=3)
    np.testing.assert_allclose(replay2.estimated, replay.estimated)
    assert replay2.metrics == {} and replay2.reference is None


def test_carmen_odometry_is_not_ground_truth(tmp_path):
    path = tmp_path / "scan.clf"
    path.write_text(
        "FLASER 4 2 2 2 2 900 800 1 0 0 0 10 host 0\n"
        "FLASER 4 2 2 2 2 800 700 2 0.1 0 0 10.1 host 0.1\n"
        "FLASER 4 2 2 2 2 700 600 3 0.2 0 0 10.2 host 0.2\n",
        encoding="utf-8",
    )
    result = replay_recording(path, ExperimentConfig(), max_frames=2)
    assert result.reference is None and result.metrics == {}
    assert result.metadata["ground_truth_from_log"] is False
    assert result.estimated[-1, 0] == pytest.approx(5.2)
    assert result.estimated[-1, 1] == pytest.approx(4)


@pytest.mark.parametrize(
    "content",
    ["FLASER 4 1\n", "FLASER 1 2 0 0 0 0 0 0 1 host 0\n", "PARAM robot_frontlaser_offset 0.2 host 0\n"],
)
def test_invalid_or_uncalibrated_carmen_is_rejected(tmp_path, content):
    path = tmp_path / "bad.clf"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_carmen(path, ExperimentConfig(), 2)


def test_reference_cannot_extrapolate(tmp_path):
    base = ExperimentConfig()
    config = replace(base, simulation=replace(base.simulation, duration_s=0.3))
    dataset = generate_dataset(config, tmp_path)
    reference = dataset / "partial_reference.csv"
    reference.write_text("timestamp_s,x_m,y_m,yaw_rad\n0,5,4,0\n0.1,5.04,4,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cover"):
        replay_recording(dataset, config, reference_path=reference, max_frames=3)


def test_carmen_sorting_is_explicit_stable_and_preserves_original(tmp_path):
    path = tmp_path / "unordered.clf"
    content = (
        "FLASER 4 2 2 2 2 900 800 1 0 0 0 10 host 0\n"
        "FLASER 4 2 2 2 2 800 700 2 0.2 0 0 10.2 host 0.2\n"
        "FLASER 4 2 2 2 2 700 600 3 0.1 0 0 10.1 host 0.1\n"
        "FLASER 4 2 2 2 2 600 500 4 99 0 0 10.1 host 0.1\n"
    )
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        load_carmen(path, ExperimentConfig(), 3)
    frames, metadata = load_carmen(path, ExperimentConfig(), 3, "sort_unique")
    assert len(frames) == 2
    assert metadata["original_temporal_inversions"] == 2
    assert metadata["duplicate_timestamps_discarded"] == 1
    assert frames[0].left_rate == pytest.approx(frames[1].left_rate)
    assert path.read_text(encoding="utf-8") == content


def test_frontier_diagnostics_are_retained_for_each_displayed_step():
    base = ExperimentConfig()
    config = replace(base, simulation=replace(base.simulation, motion_profile="frontier", duration_s=0.3))
    steps = []
    run_simulation(config, on_step=steps.append)
    assert all(
        step.navigation["navigation_pose_source"] == "estimated pose; no world geometry" for step in steps
    )
    assert steps[0].navigation is not steps[-1].navigation
