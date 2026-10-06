"""Regression checks for scientific labels and desktop failure recovery."""

import os
import time
from dataclasses import replace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

import slamcart3d.gui.app as gui_app
from slamcart3d.core.config import ExperimentConfig, SimulationConfig
from slamcart3d.experiments.sweep import run_parameter_sweep
from slamcart3d.gui.app import ExperimentWorker, MainWindow
from slamcart3d.gui.errors import localize_error
from slamcart3d.gui.robustness import RobustnessCanvas
from slamcart3d.simulation.runner import run_simulation


@pytest.fixture(scope="module")
def application():
    yield QApplication.instance() or QApplication([])


def short_config() -> ExperimentConfig:
    base = ExperimentConfig()
    return replace(
        base,
        lidar=replace(base.lidar, beam_count=8),
        simulation=SimulationConfig(duration_s=0.2),
    )


def wait_for_worker(window, application, attribute) -> None:
    deadline = time.monotonic() + 20
    while getattr(window, attribute) is not None and time.monotonic() < deadline:
        application.processEvents()
        QTest.qWait(10)
    assert getattr(window, attribute) is None


def test_sweep_labels_describe_drawn_data_not_next_form_selection(application) -> None:
    window = MainWindow()
    study = run_parameter_sweep(short_config(), "wheel_rate_noise_std_rad_s", (0.0, 0.1), (0, 1))
    window.sweep_canvas.study = study
    window._update_sweep_axis_labels()
    expected_x, expected_y = window.sweep_canvas.x_label, window.sweep_canvas.y_label
    window.sweep_parameter.setCurrentIndex(window.sweep_parameter.findData("lidar_noise_std_m"))
    window.sweep_metric.setCurrentIndex(window.sweep_metric.findData("map_f1_0p25"))
    assert (window.sweep_canvas.x_label, window.sweep_canvas.y_label) == (expected_x, expected_y)
    window.language_combo.setCurrentIndex(1)
    assert window.sweep_canvas.x_label == gui_app.TEXT["en"]["param_wheel_rate_noise_std_rad_s"]
    assert window.sweep_canvas.y_label == gui_app.METRIC_LABELS["ate_rmse_m"][1]
    window.close()


@pytest.mark.parametrize(
    "action,attribute,tab",
    [
        ("run_button", "_worker", 0),
        ("batch_button", "_batch_worker", 3),
        ("frontier_button", "_frontier_worker", 4),
        ("dataset_button", "_dataset_worker", 6),
        ("sweep_button", "_sweep_worker", 8),
    ],
)
def test_write_failure_releases_every_action(application, tmp_path, action, attribute, tab) -> None:
    window = MainWindow()
    window._set_from_config(short_config())
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("preserve this file", encoding="utf-8")
    window._inputs["output"].setText(str(blocked))
    window.dataset_output.setText(str(blocked))
    window.frontier_budget.setValue(9)
    window.frontier_seeds.setText("0")
    window.sweep_levels.setValue(2)
    window.sweep_seeds.setText("0,1")
    errors = []
    window._show_error = errors.append
    window.show()
    window.tabs.setCurrentIndex(tab)
    application.processEvents()
    QTest.mouseClick(getattr(window, action), Qt.MouseButton.LeftButton)
    wait_for_worker(window, application, attribute)
    assert errors
    for name in (
        "run_button",
        "save_button",
        "batch_button",
        "frontier_button",
        "dataset_button",
        "sweep_button",
    ):
        assert getattr(window, name).isEnabled(), name
    assert blocked.read_text(encoding="utf-8") == "preserve this file"
    assert window.statusBar().currentMessage() == gui_app.TEXT["es"]["failed"]
    window.language_combo.setCurrentIndex(1)
    status = window.metrics_hint if action == "run_button" else getattr(window, f"{action[:-7]}_status")
    assert status.text() == gui_app.TEXT["en"]["failed"]
    window._inputs["output"].setText(str(tmp_path / "recovered"))
    window.dataset_output.setText(str(tmp_path / "recovered"))
    errors.clear()
    window.tabs.setCurrentIndex(tab)
    QTest.mouseClick(getattr(window, action), Qt.MouseButton.LeftButton)
    wait_for_worker(window, application, attribute)
    assert not errors
    assert not window._failed_actions
    window.close()


def test_incremental_cleanup_failure_still_emits_original_error(monkeypatch, tmp_path) -> None:
    config = short_config()
    real_run = run_simulation

    class BrokenWriter:
        def __init__(self, started, root):
            self.target = root

        def append(self, step):
            raise OSError("original disk error")

        def close(self):
            raise OSError("flush error")

        def fail(self, detail):
            raise OSError("status error")

    monkeypatch.setattr(gui_app, "IncrementalRunWriter", BrokenWriter)
    monkeypatch.setattr(gui_app, "run_simulation", real_run)
    worker = ExperimentWorker(config, tmp_path)
    errors = []
    worker.failed.connect(errors.append)
    worker.run()
    assert len(errors) == 1
    assert all(detail in errors[0] for detail in ("original disk error", "flush error", "status error"))


@pytest.mark.parametrize("cancel", [True, False])
def test_file_dialogs_cancel_or_report_bad_input_without_losing_config(
    application, monkeypatch, tmp_path, cancel
) -> None:
    window = MainWindow()
    before = window.current_config()
    original_path = window.config_path.text()
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("experiment: {seed: 1.7}", encoding="utf-8")
    errors = []
    window._show_error = errors.append
    chosen = "" if cancel else str(invalid)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (chosen, ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: ("", ""))
    window.show()
    QTest.mouseClick(window.browse_button, Qt.MouseButton.LeftButton)
    if not cancel:
        QTest.mouseClick(window.load_button, Qt.MouseButton.LeftButton)
        assert errors and "seed" in errors[0]
    else:
        assert window.config_path.text() == original_path
    QTest.mouseClick(window.save_button, Qt.MouseButton.LeftButton)
    assert window.current_config() == before
    assert invalid.read_text(encoding="utf-8") == "experiment: {seed: 1.7}"
    window.close()


def test_dataset_with_no_camera_has_explanation_in_both_languages(application, tmp_path) -> None:
    window = MainWindow()
    window._set_from_config(short_config())
    window.dataset_output.setText(str(tmp_path))
    window._generate_dataset()
    wait_for_worker(window, application, "_dataset_worker")
    assert window.camera_preview.text() == gui_app.TEXT["es"]["dataset_no_camera"]
    window.language_combo.setCurrentIndex(1)
    assert window.camera_preview.text() == gui_app.TEXT["en"]["dataset_no_camera"]
    window.close()


@pytest.mark.parametrize("language", ["es", "en"])
def test_expected_errors_use_selected_language(application, monkeypatch, language) -> None:
    window = MainWindow()
    window.language_combo.setCurrentIndex(0 if language == "es" else 1)
    dialogs = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: dialogs.append(args[2]))
    window._show_error("empty output folder")
    window._show_error("seed must have type int")
    assert len(dialogs) == 2
    if language == "es":
        assert "Indique una carpeta" in dialogs[0]
        assert "El campo seed debe contener un entero" in dialogs[1]
    else:
        assert "empty output folder" in dialogs[0]
        assert "seed must have type int" in dialogs[1]
    window.close()


@pytest.mark.parametrize("action", ["batch", "frontier", "dataset", "sweep"])
def test_language_change_keeps_running_status_even_with_prior_results(application, monkeypatch, action):
    window = MainWindow()
    monkeypatch.setattr(window, "_show_batch_summary", lambda path: None)
    setattr(window, f"_last_{action}_path", "prior result")

    class RunningWorker:
        def isRunning(self):
            return True

    setattr(window, f"_{action}_worker", RunningWorker())
    window.language_combo.setCurrentIndex(1)
    assert getattr(window, f"{action}_status").text() == gui_app.TEXT["en"][f"{action}_running"]
    setattr(window, f"_{action}_worker", None)
    window.close()


@pytest.mark.parametrize("height", [340, 500, 800])
def test_asrf_legend_baseline_stays_inside_canvas(application, height):
    canvas = RobustnessCanvas()
    canvas.resize(600, height)
    assert canvas.plot_rectangle().bottom() + 82 <= canvas.height() - 10
    canvas.close()


def test_unknown_error_details_are_not_replaced_by_guessed_explanations():
    detail = "unexpected storage driver error 418"
    assert localize_error(detail, "es") == detail
    assert localize_error("lidar3d.enabled must have type bool", "es").endswith("true o false sin comillas.")


def test_busy_run_keeps_navigation_and_language_but_blocks_new_tasks(application, tmp_path) -> None:
    window = MainWindow()
    window._set_from_config(replace(short_config(), simulation=SimulationConfig(duration_s=0.8)))
    window._inputs["output"].setText(str(tmp_path))
    window.show()
    QTest.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
    for name in (
        "run_button",
        "save_button",
        "batch_button",
        "frontier_button",
        "dataset_button",
        "sweep_button",
    ):
        assert not getattr(window, name).isEnabled(), name
    window.language_combo.setCurrentIndex(1)
    assert window.metrics_hint.text() == gui_app.TEXT["en"]["running"]
    window.close()
    assert window.isVisible()
    wait_for_worker(window, application, "_worker")
    assert window._last_result is not None
    for index in (0, 1, window.step_slider.maximum()):
        window.step_slider.setValue(index)
        application.processEvents()
        assert window.cloud_canvas.result.timestamps[-1] == pytest.approx(index * 0.1)
    window.close()
