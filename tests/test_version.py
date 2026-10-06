"""The active release number must agree across code, packaging and CLI."""

import re
from pathlib import Path

import pytest
import yaml

from slamcart3d import __version__
from slamcart3d.cli import _parser


def test_release_metadata_is_consistent():
    root = Path(__file__).resolve().parents[1]
    assert __version__ == "1.0.0"
    package = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "1\.0\.0"$', package, re.M)
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["version"] == __version__
    assert citation["authors"][0]["email"] == "fjburgosf@gmail.com"
    assert 'email = "fjburgosf@gmail.com"' in package


def test_cli_displays_release_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        _parser().parse_args(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "SLAMCart3D 1.0.0"
