"""Bundled examples and workflows must be discoverable without file-name knowledge."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from slamcart3d.core.config import ExperimentConfig
from slamcart3d.gui import paths
from slamcart3d.gui.app import MainWindow
from slamcart3d.gui.catalog import EXAMPLE_TITLES, WORKFLOWS, available_examples


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def test_catalog_covers_every_bundled_example():
    assert {path.name for path in available_examples()} == set(EXAMPLE_TITLES)
    assert len(available_examples()) == 26
    assert len(WORKFLOWS) == 6


def test_maze_is_selectable_in_environment_and_batch_without_starting_a_task(application):
    window = MainWindow()
    try:
        combo = window._inputs["world"]
        index = combo.findData("indoor_maze")
        assert index >= 0
        previous = window.current_config()
        combo.setCurrentIndex(index)
        assert combo.currentText() == "Laberinto interior"
        assert window.current_config().simulation.world == "indoor_maze"
        assert window.current_config().simulation.motion_profile == previous.simulation.motion_profile
        assert window._worker is None and window._last_result is None
        assert "indoor_maze" in window.batch_worlds
        checkbox = window.batch_worlds["indoor_maze"]
        assert checkbox.text() == "Laberinto interior"
        checkbox.setChecked(True)
        window.language_combo.setCurrentIndex(1)
        assert combo.currentData() == "indoor_maze" and combo.currentText() == "Indoor maze"
        assert checkbox.isChecked() and checkbox.text() == "Indoor maze"
        path = paths.bundled_configs_path() / "example_indoor_maze.yaml"
        window.example_combo.setCurrentIndex(window.example_combo.findData(str(path)))
        window.load_example_button.click()
        config = window.current_config()
        assert config == ExperimentConfig.load(path)
        assert config.simulation.world == "indoor_maze"
        assert config.simulation.motion_profile == "frontier"
        assert config.lidar3d.enabled and config.occupancy_grid.enabled
        assert window._worker is None
    finally:
        window.close()


@pytest.mark.parametrize("path", available_examples(), ids=lambda path: path.stem)
def test_each_example_loads_through_visible_button(application, path):
    window = MainWindow()
    window.resize(1160, 760)
    window.show()
    try:
        index = window.example_combo.findData(str(path))
        assert index >= 0
        assert window.example_combo.itemText(index) == EXAMPLE_TITLES[path.name][0]
        previous = window.current_config()
        window.example_combo.setCurrentIndex(index)
        assert window.current_config() == previous  # Selection alone never replaces edits.
        QTest.mouseClick(window.load_example_button, Qt.MouseButton.LeftButton)
        assert window.current_config() == ExperimentConfig.load(path)
        assert window.config_path.text() == str(path)
        assert window._worker is None  # Loading never starts an experiment.
    finally:
        window.close()


@pytest.mark.parametrize("key,tab,spanish,english", WORKFLOWS)
def test_workflow_selector_navigates_without_starting_tasks(application, key, tab, spanish, english):
    window = MainWindow()
    try:
        previous = window.current_config()
        window.workflow_combo.setCurrentIndex(window.workflow_combo.findData(key))
        assert window.tabs.currentIndex() == tab
        assert window.workflow_combo.currentText() == spanish
        window.language_combo.setCurrentIndex(1)
        assert window.workflow_combo.currentData() == key
        assert window.workflow_combo.currentText() == english
        assert window.tabs.currentIndex() == tab
        assert window.current_config() == previous
        assert window._can_replay() and window.recorded_page.worker is None
        window.tabs.setCurrentIndex(0)
        assert window.workflow_combo.currentData() == "run"
    finally:
        window.close()


def test_language_changes_titles_without_loading_selected_example(application):
    window = MainWindow()
    try:
        index = window.example_combo.findData(str(paths.bundled_configs_path() / "example_custom_room.yaml"))
        window.example_combo.setCurrentIndex(index)
        previous = window.current_config()
        window.language_combo.setCurrentIndex(1)
        assert window.example_combo.currentIndex() == index
        assert window.example_combo.currentText() == "Editable custom room"
        assert window.current_config() == previous
        window.load_example_button.click()
        assert window.current_config().simulation.world == "custom"
    finally:
        window.close()


def test_reselecting_current_workflow_returns_from_results(application):
    window = MainWindow()
    try:
        window.tabs.setCurrentIndex(5)
        assert window.workflow_combo.currentData() == "run"
        window.workflow_combo.activated.emit(window.workflow_combo.currentIndex())
        assert window.tabs.currentIndex() == 0
        assert window._worker is None
    finally:
        window.close()


@pytest.mark.parametrize("worker_name", ["_worker", "recorded"])
def test_loading_is_blocked_during_work_and_restored(application, worker_name):
    window = MainWindow()
    try:
        previous = window.current_config()
        window.example_combo.setCurrentIndex(0)
        if worker_name == "recorded":
            window.recorded_page.worker = object()
        else:
            window._worker = object()
        window._set_task_buttons(False)
        assert not window.load_example_button.isEnabled()
        assert not window.load_button.isEnabled() and not window.browse_button.isEnabled()
        window._load_example()
        assert window.current_config() == previous
        window._worker = None
        window.recorded_page.worker = None
        window._set_task_buttons(True)
        assert window.load_example_button.isEnabled() and window.example_combo.isEnabled()
        window.load_example_button.click()
        assert window.current_config() == ExperimentConfig.load(available_examples()[0])
    finally:
        window._worker = None
        window.recorded_page.worker = None
        window.close()


def test_failed_example_load_preserves_form_and_path(application, monkeypatch):
    window = MainWindow()
    try:
        previous = window.current_config()
        original_path = window.config_path.text()
        errors = []
        window._show_error = errors.append

        def broken(*args):
            raise ValueError("invalid example")

        monkeypatch.setattr(ExperimentConfig, "load", broken)
        window.load_example_button.click()
        assert errors == ["invalid example"]
        assert window.current_config() == previous and window.config_path.text() == original_path
    finally:
        window.close()


def test_examples_are_located_when_launch_directory_changes(monkeypatch, tmp_path):
    expected = paths.bundled_configs_path()
    monkeypatch.chdir(tmp_path)
    assert paths.bundled_configs_path() == expected
    assert Path(paths.bundled_example_path()).is_file()
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(paths.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert paths.bundled_configs_path() == tmp_path / "configs"
