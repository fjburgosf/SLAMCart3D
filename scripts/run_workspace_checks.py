"""Run project checks with temporary files and caches inside this workspace."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    interpreter = Path(sys.executable).resolve()
    if not interpreter.is_relative_to(root / ".venv"):
        raise RuntimeError("run this script using the workspace .venv interpreter")

    runtime = root / ".tmp" / "checks" / uuid4().hex
    paths = {
        "TEMP": runtime / "temp",
        "TMP": runtime / "temp",
        "TMPDIR": runtime / "temp",
        "APPDATA": runtime / "appdata",
        "LOCALAPPDATA": runtime / "localappdata",
        "XDG_CACHE_HOME": runtime / "cache",
        "MPLCONFIGDIR": runtime / "matplotlib",
        "RUFF_CACHE_DIR": runtime / "ruff",
        "PIP_CACHE_DIR": runtime / "pip",
    }
    for directory in set(paths.values()):
        directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update({name: str(path) for name, path in paths.items()})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["QT_QPA_PLATFORM"] = "offscreen"

    commands = (
        [str(interpreter), "-m", "ruff", "check", "src", "tests", "scripts"],
        [str(interpreter), "-m", "ruff", "format", "--check", "src", "tests", "scripts"],
        [
            str(interpreter),
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--basetemp={runtime / 'pytest'}",
        ],
    )
    for command in commands:
        print("Checking:", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=root, env=environment, check=False)
        if completed.returncode:
            return completed.returncode
    print("All checks passed; runtime stayed under", runtime, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
