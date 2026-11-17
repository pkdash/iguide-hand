"""Tests for utils.setup_taudem_path (PATH handling when no --taudem-path is given)."""
import pytest


def test_setup_taudem_path_none_missing_exits(monkeypatch):
    import subprocess
    import utils

    def boom(*a, **k):
        raise FileNotFoundError("pitremove")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(SystemExit):
        utils.setup_taudem_path(None)


def test_setup_taudem_path_none_found_ok(monkeypatch):
    import subprocess
    import utils

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, b"usage", b"")
    )
    utils.setup_taudem_path(None)


def test_setup_taudem_path_uses_pathsep(monkeypatch, tmp_path):
    import os
    import utils

    monkeypatch.setenv("PATH", "/x")
    utils.setup_taudem_path(str(tmp_path))
    assert os.environ["PATH"] == f"{tmp_path}{os.pathsep}/x"
