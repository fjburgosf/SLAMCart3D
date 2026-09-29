"""Optional offscreen smoke tests for the connected desktop interface."""

import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from slamcart3d.core.config import (
    ExperimentConfig,
    LidarConfig,
    MotionSegment,
    OccupancyGridConfig,
    SimulationConfig,
)
from slamcart3d.gui import paths as gui_paths
from slamcart3d.gui.app import (
    MainWindow,
    bundled_example_path,
    default_datasets_path,
    default_results_path,
)
from slamcart3d.gui.viewer3d import PointCloudCanvas
from slamcart3d.simulation.runner import run_simulation


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    yield app


def test_language_toggle_keeps_configuration(application) -> None:
    window = MainWindow()
    precise = replace(window.current_config().simulation, linear_velocity_m_s=0.712345678)
    window._set_from_config(replace(window.current_config(), simulation=precise))
    window.language_combo.setCurrentIndex(1)
    assert window.tabs.tabText(0) == "Experiment"
    assert window.current_config().simulation.linear_velocity_m_s == pytest.approx(0.712345678)
    window._inputs["lidar3d_enabled"].setChecked(True)
    assert window.current_config().lidar3d.enabled
    assert window._inputs["lidar3d_enabled"].toolTip().startswith("Enable")
    window._inputs["camera_enabled"].setChecked(True)
    assert window.current_config().camera.enabled
    window._inputs["score_enabled"].setChecked(True)
    assert window.current_config().robustness_score.enabled
    window._inputs["occupancy_enabled"].setChecked(True)
    assert window.current_config().occupancy_grid.enabled
    window._inputs["dropout_region_enabled"].setChecked(True)
    window._inputs["dropout_region_x_min_m"].setValue(5.5)
    window._inputs["dropout_region_x_max_m"].setValue(6.2)
    assert window.current_config().degradation.dropout_region_enabled
    assert window.current_config().degradation.dropout_region_x_min_m == 5.5
    assert window._inputs["dropout_region_x_min_m"].toolTip().startswith("Inclusive")
    window.language_combo.setCurrentIndex(0)
    assert window.tabs.tabText(0) == "Experimento"
    window.close()


def test_gui_preserves_scripted_motion_segments_on_language_toggle(application) -> None:
    window = MainWindow()
    config = window.current_config()
    segments = (MotionSegment(config.simulation.duration_s, 0.4, 0.1),)
    scripted = replace(config.simulation, motion_profile="scripted", motion_segments=segments)
    window._set_from_config(replace(config, simulation=scripted))
    window.language_combo.setCurrentIndex(1)
    assert window._inputs["motion"].currentText() == "Segmented trajectory (YAML)"
    assert window.current_config().simulation.motion_segments == segments
    window.language_combo.setCurrentIndex(0)
    assert window.current_config().simulation.motion_profile == "scripted"
    window.close()


def test_occupancy_tab_renders_measured_grid_in_both_languages(application) -> None:
    window = MainWindow()
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=12),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
        occupancy_grid=OccupancyGridConfig(enabled=True),
    )
    result = run_simulation(config)
    window._on_completed(result, "test export")
    window.show()
    window.tabs.setCurrentIndex(7)
    application.processEvents()
    assert window.tabs.tabText(7) == "Ocupación 2D"
    assert window.occupancy_canvas.result is result
    assert not window.occupancy_canvas.grab().isNull()
    window.language_combo.setCurrentIndex(1)
    assert window.tabs.tabText(7) == "2D occupancy"
    window.close()


def test_frozen_default_paths_stay_next_to_executable(monkeypatch, tmp_path) -> None:
    import slamcart3d.gui.app as gui_app

    executable = tmp_path / "SLAMCart3D.exe"
    monkeypatch.setattr(gui_app.sys, "frozen", True, raising=False)
    monkeypatch.setattr(gui_app.sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(gui_app.sys, "executable", str(executable))
    assert bundled_example_path() == str(tmp_path / "configs" / "example_01_clean_environment.yaml")
    assert default_results_path() == str(tmp_path / "results")
    assert default_datasets_path() == str(tmp_path / "datasets")


def test_frozen_export_does_not_fall_back_outside_requested_folder(monkeypatch, tmp_path) -> None:
    executable = tmp_path / "app" / "SLAMCart3D.exe"
    monkeypatch.setattr(gui_paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(gui_paths.sys, "executable", str(executable))
    attempted: list[Path] = []

    def write(root: Path) -> Path:
        attempted.append(root)
        raise PermissionError("protected application directory")

    with pytest.raises(PermissionError, match="protected application directory"):
        gui_paths.export_to_requested_path(executable.parent / "results", write)
    assert attempted == [executable.parent / "results"]


def test_frozen_dataset_export_stays_in_requested_folder(monkeypatch, tmp_path) -> None:
    executable = tmp_path / "app" / "SLAMCart3D.exe"
    monkeypatch.setattr(gui_paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(gui_paths.sys, "executable", str(executable))

    def write(root: Path) -> Path:
        root.mkdir(parents=True)
        (root / "dataset.txt").write_text("ok", encoding="utf-8")
        return root

    target = gui_paths.export_to_requested_path(executable.parent / "datasets", write, kind="datasets")
    assert target == executable.parent / "datasets"
    assert (target / "dataset.txt").read_text(encoding="utf-8") == "ok"


def test_load_config_and_execute_without_freezing(application, tmp_path) -> None:
    window = MainWindow()
    config_path = Path(__file__).resolve().parents[1] / "configs" / "example_03_imu_drift.yaml"
    window.config_path.setText(str(config_path))
    window._load_config()
    assert window.current_config().slam.algorithm == "wheel_imu_fusion"
    assert window.current_config().degradation.imu_gyro_bias_rad_s == 0.03
    base = window.current_config()
    short = replace(
        base, lidar=replace(base.lidar, beam_count=8), simulation=replace(base.simulation, duration_s=0.2)
    )
    window._set_from_config(short)
    window._inputs["output"].setText(str(tmp_path))
    window._run_experiment()
    deadline = time.monotonic() + 10.0
    while window._worker is not None and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    assert window._worker is None
    assert window.run_button.isEnabled()
    assert window._last_result is not None
    assert window.metrics_table.rowCount() == len(window._last_result.metrics)
    assert len(list(tmp_path.glob("EXP-*/metadata.json"))) == 1
    window.close()


def test_batch_tab_executes_and_translates_summary(application, tmp_path) -> None:
    window = MainWindow()
    base = window.current_config()
    short = replace(
        base, lidar=replace(base.lidar, beam_count=8), simulation=replace(base.simulation, duration_s=0.2)
    )
    window._set_from_config(short)
    window._inputs["output"].setText(str(tmp_path))
    window.batch_seeds.setText("0,1")
    window._run_batch()
    deadline = time.monotonic() + 15.0
    while window._batch_worker is not None and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    assert window._batch_worker is None
    assert window._last_batch_path is not None
    assert window.batch_table.rowCount() > 0
    assert len(list(window._last_batch_path.glob("runs/EXP-*/metadata.json"))) == 2
    window.language_combo.setCurrentIndex(1)
    assert window.tabs.tabText(3) == "Batch"
    assert window.batch_table.horizontalHeaderItem(3).text() == "Mean"
    window.close()


def test_asrf_tab_executes_and_exports_map(application, tmp_path) -> None:
    window = MainWindow()
    base = window.current_config()
    short = replace(
        base, lidar=replace(base.lidar, beam_count=8), simulation=replace(base.simulation, duration_s=0.2)
    )
    window._set_from_config(short)
    window._inputs["output"].setText(str(tmp_path))
    window.frontier_seeds.setText("0,1")
    window._run_frontier()
    deadline = time.monotonic() + 15.0
    while window._frontier_worker is not None and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    assert window._frontier_worker is None
    assert window._last_frontier_path is not None
    assert window.frontier_canvas.study is not None
    assert (window._last_frontier_path / "robustness_map.csv").is_file()
    assert (window._last_frontier_path / "robustness_map_gui.png").stat().st_size > 0
    original_x_label = window.frontier_canvas.x_label
    window.frontier_x_combo.setCurrentIndex(2)
    assert window.frontier_canvas.x_label == original_x_label
    window.language_combo.setCurrentIndex(1)
    assert window.tabs.tabText(4) == "ASRF robustness"
    window.close()


def test_3d_tab_shows_estimated_cloud_and_translates(application, tmp_path) -> None:
    window = MainWindow()
    base = window.current_config()
    short = replace(
        base,
        lidar=replace(base.lidar, beam_count=8),
        lidar3d=replace(base.lidar3d, enabled=True, azimuth_beams=3, elevation_beams=3),
        simulation=replace(base.simulation, duration_s=0.2),
    )
    window._set_from_config(short)
    window._inputs["output"].setText(str(tmp_path))
    window._run_experiment()
    deadline = time.monotonic() + 10.0
    while window._worker is not None and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    assert window._last_result is not None
    assert len(window._last_result.map_points_3d) > 0
    assert isinstance(window.cloud_canvas, PointCloudCanvas)
    assert window.cloud_canvas.result is window._last_result
    assert len(list(tmp_path.glob("EXP-*/map/point_cloud_3d.csv"))) == 1
    window.tabs.setCurrentIndex(5)
    window.show()
    application.processEvents()
    before = window.cloud_canvas.azimuth_deg
    QTest.mousePress(window.cloud_canvas, Qt.MouseButton.LeftButton, pos=QPoint(80, 80))
    QTest.mouseMove(window.cloud_canvas, pos=QPoint(120, 80))
    QTest.mouseRelease(window.cloud_canvas, Qt.MouseButton.LeftButton, pos=QPoint(120, 80))
    assert window.cloud_canvas.azimuth_deg != before
    assert not window.cloud_canvas.grab().isNull()
    window.language_combo.setCurrentIndex(1)
    assert window.tabs.tabText(5) == "3D point cloud"
    assert window.cloud_canvas.count_label == "points"
    assert window.cloud_canvas.legend_labels == ("Wall", "Cloud", "Truth", "Estimate")
    assert "not 6DoF SLAM" in window.cloud_hint.text()
    window.close()


def test_asrf_gui_can_use_3d_wall_metric_when_sensor_enabled(application, tmp_path) -> None:
    window = MainWindow()
    metric_index = window.frontier_metric_combo.findData("map3d_wall_f1_0p25")
    assert metric_index >= 0
    window.frontier_metric_combo.setCurrentIndex(metric_index)
    errors = []
    window._show_error = errors.append
    window._run_frontier()
    assert errors == ["La métrica de paredes 3D requiere activar el LiDAR multicapa."]
    base = window.current_config()
    short = replace(
        base,
        lidar=replace(base.lidar, beam_count=8),
        lidar3d=replace(base.lidar3d, enabled=True, azimuth_beams=3, elevation_beams=3),
        simulation=replace(base.simulation, duration_s=0.2),
    )
    window._set_from_config(short)
    window._inputs["output"].setText(str(tmp_path))
    window.frontier_seeds.setText("0")
    window.frontier_budget.setValue(8)
    window._run_frontier()
    deadline = time.monotonic() + 15.0
    while window._frontier_worker is not None and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    assert window._frontier_worker is None
    assert window.frontier_canvas.study is not None
    assert window.frontier_canvas.study.metric_name == "map3d_wall_f1_0p25"
    assert window._last_frontier_path is not None
    assert (window._last_frontier_path / "robustness_map.csv").is_file()
    window.close()


def test_dataset_tab_generates_camera_images_and_preview(application, tmp_path) -> None:
    window = MainWindow()
    base = window.current_config()
    short = replace(
        base,
        lidar=replace(base.lidar, beam_count=8),
        camera=replace(base.camera, enabled=True, width_px=8, height_px=6),
        simulation=replace(base.simulation, duration_s=0.2),
    )
    window._set_from_config(short)
    window.dataset_output.setText(str(tmp_path))
    window._generate_dataset()
    deadline = time.monotonic() + 15.0
    while window._dataset_worker is not None and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    assert window._dataset_worker is None
    assert window._last_dataset_path is not None
    assert (window._last_dataset_path / "camera_index.csv").is_file()
    assert window.camera_preview.pixmap() is not None
    assert not window.camera_preview.pixmap().isNull()
    assert window.tabs.currentIndex() == 6
    window.language_combo.setCurrentIndex(1)
    assert window.tabs.tabText(6) == "Dataset"
    assert window.dataset_button.text() == "Generate dataset"
    window.close()


def test_srs_gui_requires_opt_in_and_displays_component_metrics(application, tmp_path) -> None:
    window = MainWindow()
    metric_index = window.frontier_metric_combo.findData("srs")
    assert metric_index >= 0
    window.frontier_metric_combo.setCurrentIndex(metric_index)
    errors = []
    window._show_error = errors.append
    window._run_frontier()
    assert errors == ["La métrica SRS requiere activar el índice SRS en el experimento."]
    base = window.current_config()
    short = replace(
        base,
        lidar=replace(base.lidar, beam_count=8),
        robustness_score=replace(base.robustness_score, enabled=True),
        simulation=replace(base.simulation, duration_s=0.2),
    )
    window._set_from_config(short)
    window._inputs["output"].setText(str(tmp_path))
    window._run_experiment()
    deadline = time.monotonic() + 10.0
    while window._worker is not None and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    assert window._last_result is not None
    assert 0.0 <= window._last_result.metrics["srs"] <= 1.0
    labels = [window.metrics_table.item(row, 0).text() for row in range(window.metrics_table.rowCount())]
    assert "Índice de robustez SRS [0–1]" in labels
    window.language_combo.setCurrentIndex(1)
    labels_en = [window.metrics_table.item(row, 0).text() for row in range(window.metrics_table.rowCount())]
    assert "SRS robustness index [0–1]" in labels_en
    window.close()


def test_each_action_button_works_through_a_mouse_click(application, monkeypatch, tmp_path) -> None:
    """Exercise the six visible actions, including both file dialogs, as a user would."""

    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    errors: list[str] = []
    window._show_error = errors.append
    example = Path(__file__).resolve().parents[1] / "configs" / "example_multilayer_lidar.yaml"
    saved = tmp_path / "saved_config.yaml"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(example), ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(saved), ""))

    def click(button) -> None:
        assert button.isVisible() and button.isEnabled(), button.text()
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        application.processEvents()

    def wait_until_done(attribute: str, seconds: float = 20.0) -> None:
        deadline = time.monotonic() + seconds
        while getattr(window, attribute) is not None and time.monotonic() < deadline:
            application.processEvents()
            QTest.qWait(10)
        assert getattr(window, attribute) is None, attribute

    click(window.browse_button)
    assert window.config_path.text() == str(example)
    click(window.load_button)
    assert window.current_config().lidar3d.enabled
    short = window.current_config()
    short = replace(
        short,
        lidar=replace(short.lidar, beam_count=8),
        lidar3d=replace(short.lidar3d, azimuth_beams=3, elevation_beams=3),
        camera=replace(short.camera, enabled=True, width_px=8, height_px=6),
        simulation=replace(short.simulation, duration_s=0.2),
    )
    window._set_from_config(short)
    window._inputs["output"].setText(str(tmp_path / "results"))
    window.dataset_output.setText(str(tmp_path / "datasets"))

    click(window.save_button)
    assert saved.is_file()
    assert ExperimentConfig.load(saved).lidar3d.enabled
    click(window.load_button)
    assert window.current_config().camera.enabled

    click(window.run_button)
    wait_until_done("_worker")
    assert window._last_result is not None
    assert window.canvas.result is window._last_result
    assert window.cloud_canvas.result is window._last_result
    assert window.metrics_table.rowCount() > 0

    window.tabs.setCurrentIndex(3)
    window.batch_seeds.setText("0,1")
    click(window.batch_button)
    wait_until_done("_batch_worker")
    assert window._last_batch_path is not None
    assert window.batch_table.rowCount() > 0

    window.tabs.setCurrentIndex(4)
    window.frontier_seeds.setText("0")
    window.frontier_budget.setValue(9)
    click(window.frontier_button)
    wait_until_done("_frontier_worker")
    assert window._last_frontier_path is not None
    assert window.frontier_canvas.study is not None

    window.tabs.setCurrentIndex(6)
    click(window.dataset_button)
    wait_until_done("_dataset_worker")
    assert window._last_dataset_path is not None
    assert window.camera_preview.pixmap() is not None
    assert errors == []
    window.close()
