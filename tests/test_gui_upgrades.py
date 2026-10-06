"""Clicks, bilingual feedback and recovery for the new scientific controls."""

import time
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from slamcart3d.core.config import ExperimentConfig
from slamcart3d.datasets.generator import generate_dataset
from slamcart3d.gui.app import MainWindow
from slamcart3d.simulation.runner import run_simulation


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def short_config():
    base = ExperimentConfig()
    return replace(
        base,
        simulation=replace(base.simulation, duration_s=0.3),
        lidar=replace(base.lidar, beam_count=12),
        lidar3d=replace(base.lidar3d, enabled=True, azimuth_beams=6, elevation_beams=3),
        occupancy_grid=replace(base.occupancy_grid, enabled=True),
    )


def wait(window, application):
    deadline = time.monotonic() + 20
    while window.recorded_page.worker is not None and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    assert window.recorded_page.worker is None


def test_prepare_exploration_and_thresholds_preserve_language(application):
    window = MainWindow()
    window.show()
    window._inputs["position_limit"].setValue(0.2)
    QTest.mouseClick(window.explore_button, Qt.MouseButton.LeftButton)
    config = window.current_config()
    assert config.simulation.motion_profile == "frontier"
    assert config.slam.algorithm == "posegraph_hybrid"
    assert config.lidar3d.enabled and config.occupancy_grid.enabled
    assert config.evaluation.position_limit_m == 0.2
    window.language_combo.setCurrentIndex(1)
    assert window.current_config() == config
    assert window.tabs.tabText(9) == "Recorded data"
    window.close()


def test_live_assessment_and_observation_controls_do_not_change_results(application):
    window = MainWindow()
    result = run_simulation(short_config())
    estimate = result.estimated.copy()
    estimate[-1, 0] += 1
    result = replace(result, estimated=estimate)
    window._on_completed(result, "test result")
    window.show()
    window.tabs.setCurrentIndex(5)
    assert "supera el límite" in window.quality_panel.label.text()
    assert "Cobertura" in window.quality_panel.note.text()
    before = result.map_points_3d.copy()
    for checkbox in (window.cloud_observed_checkbox, window.cloud_error_checkbox):
        QTest.mouseClick(checkbox, Qt.MouseButton.LeftButton)
    assert window.cloud_canvas.show_observation and window.cloud_canvas.color_error
    assert not window.cloud_canvas.grab().isNull()
    np.testing.assert_array_equal(before, result.map_points_3d)
    window.language_combo.setCurrentIndex(1)
    assert "exceeds limit" in window.quality_panel.label.text()
    assert "Coverage" in window.quality_panel.note.text()
    window.close()


def test_recorded_buttons_cancel_browse_and_replay(application, monkeypatch, tmp_path):
    window = MainWindow()
    window._set_from_config(short_config())
    dataset = generate_dataset(short_config(), tmp_path)
    window._inputs["output"].setText(str(tmp_path / "replays"))
    window.show()
    window.tabs.setCurrentIndex(9)
    page = window.recorded_page
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    page.source.setText("preserved")
    QTest.mouseClick(page.file_button, Qt.MouseButton.LeftButton)
    assert page.source.text() == "preserved"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(dataset))
    QTest.mouseClick(page.folder_button, Qt.MouseButton.LeftButton)
    assert page.source.text() == str(dataset)
    reference = str(dataset / "ground_truth.csv")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (reference, ""))
    QTest.mouseClick(page.reference_button, Qt.MouseButton.LeftButton)
    assert page.reference.text() == reference
    errors = []
    window._show_error = errors.append
    QTest.mouseClick(page.run_button, Qt.MouseButton.LeftButton)
    assert not window.run_button.isEnabled() and not page.run_button.isEnabled()
    window.close()
    assert window.isVisible()
    wait(window, application)
    assert not errors and page.last_result.reference is not None
    assert window.run_button.isEnabled() and page.run_button.isEnabled()
    assert "ATE" in page.status.text()
    window.close()


def test_recorded_write_failure_preserves_file_and_recovers(application, tmp_path):
    config = short_config()
    dataset = generate_dataset(config, tmp_path)
    blocked = tmp_path / "existing_file"
    blocked.write_text("preserved", encoding="utf-8")
    window = MainWindow()
    window._set_from_config(config)
    window.recorded_page.source.setText(str(dataset))
    window._inputs["output"].setText(str(blocked))
    errors = []
    window._show_error = errors.append
    window.recorded_page.run_button.click()
    wait(window, application)
    assert errors and blocked.read_text() == "preserved"
    window.language_combo.setCurrentIndex(1)
    assert "failed" in window.recorded_page.status.text()
    window._inputs["output"].setText(str(tmp_path / "recovered"))
    errors.clear()
    window.recorded_page.run_button.click()
    wait(window, application)
    assert not errors and window.recorded_page.last_result.reference is None
    assert "Accuracy not evaluated" in window.recorded_page.status.text()
    window.close()
