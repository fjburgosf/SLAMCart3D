"""Check the desktop launch contract and workspace-local runtime directories."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from slamcart3d.gui.paths import configure_local_runtime


def test_runtime_directories_stay_inside_launch_folder(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    names = (
        "TEMP",
        "TMP",
        "TMPDIR",
        "APPDATA",
        "LOCALAPPDATA",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "MPLCONFIGDIR",
    )
    for name in names:
        monkeypatch.setenv(name, str(tmp_path))
    runtime = configure_local_runtime()
    assert runtime == tmp_path / ".runtime"
    for name in names:
        target = Path(os.environ[name])
        assert target.is_relative_to(runtime) and target.is_dir()


def test_desktop_self_test_covers_current_tabs_and_estimators(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    shutil.copy2(root / "configs" / "example_multilayer_lidar.yaml", tmp_path / "configs")
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "slamcart3d.gui.app", "--self-test"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "results").exists()
    assert (tmp_path / ".runtime" / "temp").is_dir()
