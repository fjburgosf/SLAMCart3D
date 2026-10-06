"""Metric world geometry, camera orientation and connected view controls."""

import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from slamcart3d.core.config import ExperimentConfig, LidarConfig, RobotConfig, SimulationConfig
from slamcart3d.gui.app import MainWindow
from slamcart3d.gui.viewer3d import PointCloudCanvas
from slamcart3d.robots.geometry import cart_clearance_radius, cart_vertices, wheel_vertices
from slamcart3d.simulation.runner import run_simulation


@pytest.fixture(scope="module")
def application():
    yield QApplication.instance() or QApplication([])


@pytest.mark.parametrize("azimuth", [0, 35, 90, 180, 270])
@pytest.mark.parametrize("perspective", [False, True])
def test_positive_height_projects_up_not_below_floor(application, azimuth, perspective) -> None:
    canvas = PointCloudCanvas()
    canvas.azimuth_deg = azimuth
    canvas.perspective = perspective
    projected, _ = canvas._project(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]), np.zeros(3))
    assert projected[1, 1] > projected[0, 1]
    # paintEvent inverts screen Y exactly once: the wall top is above its base.
    assert -projected[1, 1] < -projected[0, 1]


@pytest.mark.parametrize("radius,base", [(0.1, 0.5), (0.2, 0.3), (0.05, 1.0)])
def test_wheels_touch_floor_and_shared_circle_encloses_cart(radius, base) -> None:
    for yaw in np.linspace(0, 2 * np.pi, 9):
        pose = np.array([2.3, -1.7, yaw])
        chassis = cart_vertices(pose, radius, base)
        wheels = [wheel_vertices(pose, side, radius, base) for side in (-1, 1)]
        for wheel in wheels:
            assert wheel[:, 2].min() == pytest.approx(0, abs=1e-12)
            assert wheel[:, 2].max() == pytest.approx(2 * radius)
        points = np.vstack((chassis, *wheels))
        assert (
            np.linalg.norm(points[:, :2] - pose[:2], axis=1).max()
            <= cart_clearance_radius(radius, base) + 1e-12
        )
        axle_centers = [wheel[:, :2].mean(axis=0) for wheel in wheels]
        assert np.linalg.norm(axle_centers[1] - axle_centers[0]) == pytest.approx(base)


def test_stop_response_protects_displayed_body_not_only_center() -> None:
    config = ExperimentConfig(
        robot=RobotConfig(initial_x_m=8.0, initial_y_m=4.0),
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(duration_s=10, yaw_rate_rad_s=0),
    )
    result = run_simulation(config)
    assert result.metrics["wall_blocked_steps"] > 0
    for pose in result.ground_truth:
        vertices = np.vstack((cart_vertices(pose), wheel_vertices(pose, -1), wheel_vertices(pose, 1)))
        assert vertices[:, 0].max() < 10


def test_view_controls_are_bilingual_and_do_not_mutate_measurements(application) -> None:
    window = MainWindow()
    assert window.current_config().lidar3d.enabled
    assert window.current_config() == ExperimentConfig.load(window.config_path.text())
    config = replace(window.current_config(), simulation=SimulationConfig(duration_s=0.2))
    result = run_simulation(config)
    original = result.map_points.copy()
    window._on_completed(result, "test")
    window.tabs.setCurrentIndex(5)
    window.show()
    application.processEvents()
    QTest.mouseClick(window.cloud_top_button, Qt.MouseButton.LeftButton)
    assert window.cloud_canvas.elevation_deg == 90
    QTest.mouseClick(window.cloud_reset_button, Qt.MouseButton.LeftButton)
    assert window.cloud_canvas.elevation_deg == 38
    for checkbox, option in (
        (window.cloud_cutaway_checkbox, "cutaway"),
        (window.cloud_points_checkbox, "show_cloud"),
        (window.cloud_perspective_checkbox, "perspective"),
    ):
        QTest.mouseClick(checkbox, Qt.MouseButton.LeftButton, pos=QPoint(8, checkbox.height() // 2))
        assert not getattr(window.cloud_canvas, option)
    window.language_combo.setCurrentIndex(1)
    assert window.cloud_top_button.text() == "Top view"
    assert window.cloud_cutaway_checkbox.text() == "Wall cutaway"
    assert not window.cloud_canvas.cutaway
    assert np.array_equal(original, result.map_points)
    assert window.cloud_canvas.result is result
    assert not window.cloud_canvas.grab().isNull()
    window.close()


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
def test_all_worlds_render_full_cutaway_top_and_estimation_views(application, world) -> None:
    canvas = PointCloudCanvas()
    canvas.resize(900, 600)
    canvas.result = run_simulation(
        ExperimentConfig(
            lidar=LidarConfig(beam_count=8), simulation=SimulationConfig(world=world, duration_s=0.1)
        )
    )
    canvas.show()
    for reference, cutaway, elevation in (
        (True, True, 38),
        (True, False, 20),
        (True, False, 90),
        (False, False, 38),
    ):
        canvas.show_reference, canvas.cutaway, canvas.elevation_deg = reference, cutaway, elevation
        image = canvas.grab().toImage()
        assert not image.isNull()
        assert image.width() == 900
    canvas.close()
