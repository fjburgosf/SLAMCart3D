"""Measured 3D localization, voxel evidence, surfaces and temporal GUI views."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest

from slamcart3d.core.config import ExperimentConfig, ReconstructionConfig
from slamcart3d.core.models import LidarScan3D, Pose2D
from slamcart3d.io.export import IncrementalRunWriter, export_result
from slamcart3d.mapping.voxel import VoxelMap, pose_matrix, transform_points
from slamcart3d.simulation.runner import run_simulation
from slamcart3d.slam.factory import make_algorithm
from slamcart3d.slam.hybrid3d import local_points, measured_normals, register_xyz


def config_3d():
    return ExperimentConfig.load("configs/example_hybrid_3d.yaml")


def test_pose_transform_is_rigid_and_uses_all_six_components():
    state = np.array([1, 2, 3, 0.1, -0.2, 0.3])
    matrix = pose_matrix(state)
    assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3))
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1)
    assert np.allclose(transform_points(np.zeros((1, 3)), state)[0], state[:3])


def test_xyz_registration_recovers_nonplanar_pose_without_a_world():
    source = np.random.default_rng(17).uniform(-2, 2, (180, 3))
    state = np.array([0.06, -0.04, 0.05, 0.025, -0.02, 0.035])
    target = transform_points(source, state)
    estimate, rms, fraction = register_xyz(source, target, np.zeros(6), np.ones(6) * 0.01, iterations=15)
    assert fraction == 1 and rms < 0.0001
    assert np.allclose(estimate, state, atol=1e-4)


def test_measured_plane_does_not_invent_tangential_motion():
    x, y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    target = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    normals = measured_normals(target)
    assert np.allclose(np.abs(normals[:, 2]), 1)
    source = target + [0.07, -0.04, 0.05]
    estimate, rms, fraction = register_xyz(source, target, np.zeros(6), np.ones(6) * 0.1, normals=normals)
    assert fraction > 0.9 and rms < 0.001
    assert estimate[0] == pytest.approx(0, abs=1e-8)
    assert estimate[1] == pytest.approx(0, abs=1e-8)
    assert estimate[2] == pytest.approx(-0.05, abs=0.001)


def test_unsupported_surfaces_and_missing_normals_cause_fallback():
    points = np.arange(60).reshape(20, 3).astype(float)
    assert not np.any(measured_normals(points))
    estimate, rms, fraction = register_xyz(
        points, points, np.zeros(6), np.ones(6), normals=np.zeros_like(points)
    )
    assert estimate is None and np.isinf(rms) and fraction == 0
    with pytest.raises(ValueError, match="normals"):
        register_xyz(points, points, np.zeros(6), np.ones(6), normals=np.ones((5, 3)))


@pytest.mark.parametrize("weights", [np.zeros(6), np.full(6, np.nan), np.ones(5)])
def test_registration_rejects_invalid_prior(weights):
    with pytest.raises(ValueError):
        register_xyz(np.ones((20, 3)), np.ones((20, 3)), np.zeros(6), weights)


def test_missing_3d_returns_do_not_create_geometry():
    scan = LidarScan3D(0, np.zeros(15), np.zeros(15), np.full(15, np.nan))
    assert local_points(scan, 0.5, 12).shape == (0, 3)
    algorithm = make_algorithm(config_3d())
    algorithm.initialize(Pose2D(0, 0, 0))
    pose = algorithm.update_3d(0, 1, 1, 0.1, None, None, scan)
    assert pose.x == pytest.approx(0.01)
    assert len(algorithm.xyz_chunks) == 0 and len(algorithm.get_map()) == 0
    with pytest.raises(ValueError, match="multilayer"):
        algorithm.update(0, 0, 0, 0.1, None)
    algorithm.reset()
    with pytest.raises(RuntimeError):
        algorithm.get_pose()


def test_endpoint_counts_are_per_scan_and_missing_rays_leave_unknown():
    voxels = VoxelMap(0.25, 2)
    origin, points = np.array([0.1, 0.1, 0.1]), np.array([[1.1, 0.1, 0.1]] * 3)
    voxels.observe(origin, points)
    assert voxels.hits == {(4, 0, 0): 1} and not voxels.occupied
    assert (4, 0, 0) not in voxels.free and (0, 0, 0) in voxels.free
    voxels.observe(origin, np.empty((0, 3)))
    assert voxels.hits == {(4, 0, 0): 1}
    voxels.observe(origin, points)
    assert voxels.occupied == {(4, 0, 0)}
    assert (100, 0, 0) not in voxels.free and (100, 0, 0) not in voxels.hits


def test_mesh_has_only_measured_boundary_faces_with_outward_normals():
    voxels = VoxelMap(0.25, 1)
    voxels.hits = {(0, 0, 0): 1}
    mesh = voxels.surface()
    assert len(mesh.vertices) == 8 and len(mesh.triangles) == 12
    for triangle in mesh.triangles:
        points = mesh.vertices[triangle]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        assert np.dot(normal, points.mean(axis=0) - 0.125) > 0
    voxels.hits[(1, 0, 0)] = 1
    mesh = voxels.surface()
    assert len(mesh.triangles) == 20  # shared face is not exported twice
    assert mesh.vertices[:, 0].max() == 0.5


@pytest.mark.parametrize(
    "data",
    [
        dict(resolution_m=0),
        dict(resolution_m=float("nan")),
        dict(min_hits=True),
        dict(min_hits=0),
        dict(enabled="yes"),
    ],
)
def test_invalid_reconstruction_configuration(data):
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"reconstruction": data})


def test_hybrid_3d_requires_the_sensor():
    with pytest.raises(ValueError, match="LiDAR"):
        ExperimentConfig.from_dict({"slam": {"algorithm": "hybrid_3d"}})


def test_simulation_exports_measured_surface_and_incremental_pose(tmp_path):
    base = config_3d()
    config = replace(base, simulation=replace(base.simulation, duration_s=1, motion_profile="constant"))
    writers, steps = [], []

    def start(value):
        writers.append(IncrementalRunWriter(value, tmp_path / "live"))

    def step(value):
        steps.append(value)
        writers[0].append(value)

    result = run_simulation(config, on_start=start, on_step=step)
    writers[0].complete()
    writers[0].close()
    assert result.estimated_3d.shape == (11, 6) and np.isfinite(result.estimated_3d).all()
    assert result.diagnostics["accepted_map_registrations"] > 0
    assert result.voxel_map.occupied and result.metrics["reconstruction_free_voxels"] > 0
    assert result.diagnostics["registration_input"] == "measured XYZ multilayer returns"
    assert steps[0].voxel_map is None and steps[-1].voxel_map is not result.voxel_map
    path = export_result(result, tmp_path / "finished")
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["reconstruction"]["world_reference_used"] is False
    mesh = result.voxel_map.surface()
    ply = (path / "map/reconstructed_surface.ply").read_text()
    assert f"element vertex {len(mesh.vertices)}" in ply
    assert f"element face {len(mesh.triangles)}" in ply and "end_header" in ply
    assert "occupied" in (path / "map/voxel_grid_3d.csv").read_text()
    assert np.loadtxt(path / "trajectory_3d.csv", delimiter=",", skiprows=1).shape == (11, 7)
    assert len((writers[0].target / "trajectory_3d_live.csv").read_text().splitlines()) == 11


def test_surface_can_be_disabled_independently_of_3d_localization():
    base = config_3d()
    config = replace(
        base,
        simulation=replace(base.simulation, duration_s=0.2),
        reconstruction=ReconstructionConfig(enabled=False),
    )
    result = run_simulation(config)
    assert result.voxel_map is None and len(result.map_points_3d) > 0


def test_surface_buttons_and_timeline_never_expose_future_measurements():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from slamcart3d.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    base = config_3d()
    config = replace(base, simulation=replace(base.simulation, duration_s=1.2))
    window = MainWindow()
    try:
        window._set_from_config(config)
        assert window.current_config() == config
        result = run_simulation(
            config,
            on_start=lambda value: window._on_started_run(value, "test"),
            on_step=window._on_step_ready,
        )
        window._on_completed(result, "test")
        window.step_slider.setValue(0)
        assert not window.cloud_canvas.result.voxel_map.hits
        window.step_slider.setValue(5)
        assert window.cloud_canvas.result.estimated_3d.shape == (6, 6)
        assert max(window.cloud_canvas.result.voxel_map.hits.values()) <= 5
        window.step_slider.setValue(11)
        assert max(window.cloud_canvas.result.voxel_map.hits.values()) <= 11
        window.surface_only_button.click()
        assert not window.cloud_canvas.show_reference and not window.cloud_canvas.show_cloud
        assert window.cloud_canvas.show_surface
        window.surface_compare_button.click()
        assert window.cloud_canvas.show_reference
        window.cloud_surface_checkbox.click()
        assert not window.cloud_canvas.show_surface
        window.language_combo.setCurrentIndex(1)
        assert window.surface_only_button.text() == "Reconstruction only"
        assert "level-ground" in window.cloud_hint.text()
        window.show()
        app.processEvents()
        assert not window.cloud_canvas.grab().isNull()
    finally:
        window.close()
