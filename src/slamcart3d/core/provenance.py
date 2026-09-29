"""Content fingerprint for exact source-tree provenance in exported data."""

from __future__ import annotations

import re
from functools import lru_cache
from hashlib import sha256
from pathlib import Path


@lru_cache(maxsize=1)
def source_sha256() -> str:
    """Hash every package Python source path and its bytes in stable order."""

    package_root = Path(__file__).resolve().parents[1]
    sources = sorted(package_root.rglob("*.py"))
    if not sources:
        # A frozen distribution contains bytecode, not the original .py files.
        # Its build script bundles the exact source-tree digest as immutable data.
        manifest = package_root / "source_sha256.txt"
        if not manifest.is_file():
            raise RuntimeError("package source fingerprint is unavailable")
        fingerprint = manifest.read_text(encoding="ascii").strip()
        if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
            raise RuntimeError("invalid bundled source fingerprint")
        return fingerprint
    digest = sha256()
    for source in sources:
        digest.update(source.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
