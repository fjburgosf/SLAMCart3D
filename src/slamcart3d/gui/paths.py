"""Frozen resource locations and explicit output paths for the desktop GUI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable


def bundled_example_path() -> str:
    """Locate the example in either the source checkout or the frozen bundle."""

    if getattr(sys, "frozen", False):
        return str(Path(sys._MEIPASS) / "configs" / "example_01_clean_environment.yaml")
    return "configs/example_01_clean_environment.yaml"


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
