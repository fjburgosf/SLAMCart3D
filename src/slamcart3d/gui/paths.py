"""Frozen resource locations and explicit output paths for the desktop GUI."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable


def configure_local_runtime() -> Path:
    """Keep runtime caches and temporary data beside the application or checkout."""

    root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()
    runtime = root / ".runtime"
    locations = {
        "TEMP": runtime / "temp",
        "TMP": runtime / "temp",
        "TMPDIR": runtime / "temp",
        "APPDATA": runtime / "appdata",
        "LOCALAPPDATA": runtime / "localappdata",
        "XDG_CACHE_HOME": runtime / "cache",
        "XDG_CONFIG_HOME": runtime / "config",
        "MPLCONFIGDIR": runtime / "matplotlib",
    }
    for path in set(locations.values()):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update({name: str(path) for name, path in locations.items()})
    return runtime


def bundled_configs_path() -> Path:
    """Locate examples independently of the launcher's working directory."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "configs"
    return Path(__file__).resolve().parents[3] / "configs"


def bundled_example_path() -> str:
    """Locate the default example in the checkout or the frozen bundle."""
    return str(bundled_configs_path() / "example_multilayer_lidar.yaml")


def default_results_path() -> str:
    """Keep a frozen application's first output attempt beside its executable."""

    if getattr(sys, "frozen", False):
        return str(Path(sys.executable).resolve().parent / "results")
    return "results"


def default_datasets_path() -> str:
    """Keep datasets beside a frozen application when the folder is writable."""

    if getattr(sys, "frozen", False):
        return str(Path(sys.executable).resolve().parent / "datasets")
    return "datasets"


def export_to_requested_path(
    output_root: Path, writer: Callable[[Path], Path], kind: str = "results"
) -> Path:
    """Export only to the requested folder; never silently redirect output."""

    if kind not in {"results", "datasets"}:
        raise ValueError("export kind must be results or datasets")
    return writer(output_root)
