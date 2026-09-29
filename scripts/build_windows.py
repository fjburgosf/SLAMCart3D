"""Create an isolated development Windows GUI build inside this checkout."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from slamcart3d.core.provenance import source_sha256


def main() -> int:
    if sys.platform != "win32":
        raise RuntimeError("this build script is for Windows only")
    root = Path(__file__).resolve().parents[1]
    interpreter = Path(sys.executable).resolve()
    if not interpreter.is_relative_to(root / ".venv"):
        raise RuntimeError("run this script using the workspace .venv interpreter")

    console = "--console" in sys.argv[1:]
    if any(argument != "--console" for argument in sys.argv[1:]):
        raise ValueError("the only supported option is --console")
    build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    work = root / "build" / build_id
    dist = root / "dist" / build_id
    temporary = root / ".tmp" / "build" / build_id
    for directory in (work, dist, temporary):
        directory.mkdir(parents=True, exist_ok=False)
    manifest = temporary / "source_sha256.txt"
    manifest.write_text(source_sha256() + "\n", encoding="ascii")

    env = os.environ.copy()
    env["TEMP"] = str(temporary)
    env["TMP"] = str(temporary)
    env["TMPDIR"] = str(temporary)
    env["PYINSTALLER_CONFIG_DIR"] = str(temporary / "pyinstaller-config")
    env["PIP_CACHE_DIR"] = str(temporary / "pip-cache")
    env["XDG_CACHE_HOME"] = str(temporary / "cache")
    env["APPDATA"] = str(temporary / "roaming")
    env["LOCALAPPDATA"] = str(temporary / "local")
    windows_root = Path(os.environ["SystemRoot"])
    env["PATH"] = os.pathsep.join(
        (str(interpreter.parent), str(windows_root / "System32"), str(windows_root))
    )
    command = [
        str(interpreter),
        "-m",
        "PyInstaller",
        "--onedir",
        "--console" if console else "--windowed",
        "--noupx",
        "--log-level",
        "WARN",
        "--name",
        "SLAMCart3D",
        "--paths",
        str(root / "src"),
        "--distpath",
        str(dist),
        "--workpath",
        str(work / "work"),
        "--specpath",
        str(work),
        "--add-data",
        f"{manifest};slamcart3d",
        "--add-data",
        f"{root / 'configs'};configs",
        str(root / "src" / "slamcart3d" / "gui" / "app.py"),
    ]
    print(f"Building {build_id} inside {root}", flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)
    executable = dist / "SLAMCart3D" / "SLAMCart3D.exe"
    if not executable.is_file():
        raise RuntimeError(f"expected executable missing: {executable}")
    print(f"Development build: {executable}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
