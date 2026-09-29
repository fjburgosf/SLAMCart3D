"""Check source provenance in editable and frozen-like layouts."""

import pytest

from slamcart3d.core import provenance


def test_source_fingerprint_is_stable() -> None:
    fingerprint = provenance.source_sha256()
    assert len(fingerprint) == 64
    assert fingerprint == provenance.source_sha256()


def test_bundled_fingerprint_fallback(monkeypatch, tmp_path) -> None:
    package_root = tmp_path / "slamcart3d"
    package_root.mkdir()
    module_path = package_root / "core" / "provenance.pyc"
    monkeypatch.setattr(provenance, "__file__", str(module_path))
    provenance.source_sha256.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="unavailable"):
            provenance.source_sha256()
        (package_root / "source_sha256.txt").write_text("a" * 64 + "\n", encoding="ascii")
        assert provenance.source_sha256() == "a" * 64
        provenance.source_sha256.cache_clear()
        (package_root / "source_sha256.txt").write_text("not a hash\n", encoding="ascii")
        with pytest.raises(RuntimeError, match="invalid"):
            provenance.source_sha256()
    finally:
        provenance.source_sha256.cache_clear()
