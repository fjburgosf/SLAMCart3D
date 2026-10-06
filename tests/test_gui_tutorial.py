"""The integrated guide must navigate real controls without running or losing edits."""

import os
import time
from dataclasses import replace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from slamcart3d.core.config import ExperimentConfig
from slamcart3d.gui.app import MainWindow
from slamcart3d.gui.tutorial import STEPS


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("index,step", tuple(enumerate(STEPS)))
def test_steps_visit_real_pages_without_replacing_edits(application, index, step):
    window = MainWindow()
    window.show()
    try:
        original = window.current_config()
        QTest.mouseClick(window.tutorial_button, Qt.MouseButton.LeftButton)
        assert not window.tutorial.isHidden()
        guide = window.tutorial
        guide.step_combo.setCurrentIndex(index)
        QTest.mouseClick(guide.visit_button, Qt.MouseButton.LeftButton)
        assert window.tabs.currentIndex() == step[0]
        assert guide.content.text() == step[3]
        assert window.current_config() == original and window._worker is None
        window.language_combo.setCurrentIndex(1)
        assert guide.index == index and guide.content.text() == step[4]
        assert guide.visit_button.text() == "Open: " + window.tabs.tabText(step[0])
        assert window.current_config() == original
    finally:
        window.close()


def test_f1_reopens_and_finishing_preserves_progress(application):
    window = MainWindow()
    window.show()
    try:
        window.activateWindow()
        window.setFocus()
        application.processEvents()
        QTest.qWait(20)
        QTest.keyClick(window, Qt.Key.Key_F1)
        application.processEvents()
        assert not window.tutorial.isHidden()
        guide = window.tutorial
        assert guide.index == 0 and not guide.previous_button.isEnabled()
        guide.next_button.click()
        assert guide.index == 1 and guide.prepare_button.isVisible()
        guide.previous_button.click()
        assert guide.index == 0
        guide.step_combo.setCurrentIndex(len(STEPS) - 1)
        guide.next_button.click()
        assert guide.isHidden()
        window.tutorial_button.click()
        assert guide.index == len(STEPS) - 1 and not guide.isHidden()
        guide.hide()
        assert not guide.timer.isActive()
    finally:
        window.close()


@pytest.mark.parametrize("answer", [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes])
def test_practice_requires_confirmation_preserves_output_and_does_not_run(application, monkeypatch, answer):
    window = MainWindow()
    try:
        original = window.current_config()
        original_path = window.config_path.text()
        original_output = window._inputs["output"].text()
        calls = []

        def question(*args):
            calls.append(args)
            return answer

        monkeypatch.setattr(QMessageBox, "question", question)
        window.tutorial.step_combo.setCurrentIndex(1)
        window.tutorial.prepare_button.click()
        assert len(calls) == 1 and calls[0][-1] == QMessageBox.StandardButton.No
        assert window._inputs["output"].text() == original_output and window._worker is None
        if answer == QMessageBox.StandardButton.No:
            assert window.current_config() == original and window.config_path.text() == original_path
            assert window.tutorial.index == 1
        else:
            config = window.current_config()
            assert config.name == "tutorial_exploration" and config.seed == 42
            assert config.simulation.duration_s == 30 and config.simulation.motion_profile == "frontier"
            assert config.slam.algorithm == "adaptive_hybrid"
            assert config.lidar3d.enabled and config.occupancy_grid.enabled
            assert window.tutorial.index == 2 and window.tabs.currentIndex() == 0
    finally:
        window.close()


@pytest.mark.parametrize(
    "worker_name",
    ["_worker", "_batch_worker", "_frontier_worker", "_dataset_worker", "_sweep_worker", "recorded"],
)
def test_reading_allowed_but_practice_blocked_during_every_task(application, monkeypatch, worker_name):
    window = MainWindow()
    try:
        original = window.current_config()
        if worker_name == "recorded":
            window.recorded_page.worker = object()
        else:
            setattr(window, worker_name, object())
        monkeypatch.setattr(QMessageBox, "question", lambda *args: pytest.fail("unexpected confirmation"))
        window._set_task_buttons(False)
        window.tutorial_button.click()
        assert not window.tutorial.isHidden()
        assert not window.tutorial.prepare_button.isEnabled()
        assert not window._prepare_tutorial_practice()
        assert window.current_config() == original
        assert "Cálculo en curso" in window.tutorial.state_label.text()
    finally:
        for name in ("_worker", "_batch_worker", "_frontier_worker", "_dataset_worker", "_sweep_worker"):
            setattr(window, name, None)
        window.recorded_page.worker = None
        window._set_task_buttons(True)
        assert window.tutorial.prepare_button.isEnabled()
        window.close()


def test_failed_practice_load_preserves_form(application, monkeypatch):
    window = MainWindow()
    try:
        original = window.current_config()
        original_path = window.config_path.text()
        errors = []
        window._show_error = errors.append
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)

        def broken(*args):
            raise ValueError("missing practice")

        monkeypatch.setattr(ExperimentConfig, "load", broken)
        assert not window._prepare_tutorial_practice()
        assert errors == ["missing practice"]
        assert window.current_config() == original and window.config_path.text() == original_path
    finally:
        window.close()


def test_confirmed_practice_executes_and_guide_remains_usable(application, monkeypatch, tmp_path):
    window = MainWindow()
    try:
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
        window._inputs["output"].setText(str(tmp_path / "results"))
        assert window._prepare_tutorial_practice()
        config = window.current_config()
        # Shorten only the regression run; the delivered practice remains 30 simulated seconds.
        window._set_from_config(replace(config, simulation=replace(config.simulation, duration_s=0.3)))
        window.tutorial_button.click()
        window.run_button.click()
        deadline = time.monotonic() + 30
        while window._worker is not None and time.monotonic() < deadline:
            application.processEvents()
            QTest.qWait(10)
        assert window._worker is None and window._last_result is not None
        window.tutorial.refresh_state()
        assert "corrida terminada" in window.tutorial.state_label.text()
        assert window.metrics_table.rowCount() > 0
        assert list((tmp_path / "results").glob("EXP-*/metrics.csv"))
        window.tutorial.step_combo.setCurrentIndex(5)
        window.tutorial.visit_button.click()
        assert window.tabs.currentIndex() == 2
    finally:
        if window._worker is not None:
            window._worker.wait(30000)
            application.processEvents()
        window.close()
