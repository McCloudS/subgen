"""launcher.py installs the requirements of the branch it runs (dev used to get main's, with stable-ts)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

import launcher


@pytest.mark.parametrize("args,env_branch,expected", [
    (["-i", "-b", "dev", "-x"], None, "/dev/requirements.txt"),
    (["-i", "-x"], "dev", "/dev/requirements.txt"),   # BRANCH env, as in the dev Docker images
    (["-i", "-x"], None, "/main/requirements.txt"),
])
def test_install_uses_branch_requirements(monkeypatch, args, env_branch, expected):
    urls = []
    monkeypatch.setattr(launcher, "download_from_github", lambda url, out: urls.append(url))
    monkeypatch.setattr(launcher, "install_packages_from_requirements", lambda f: None)
    monkeypatch.setattr(launcher, "load_env_variables", lambda *a, **k: None)
    monkeypatch.setattr(launcher.os, "chdir", lambda *a: None)
    if env_branch:
        monkeypatch.setenv("BRANCH", env_branch)
    else:
        monkeypatch.delenv("BRANCH", raising=False)
    monkeypatch.setattr(sys, "argv", ["launcher.py"] + args)
    launcher.main()
    assert any(u.endswith(expected) for u in urls), urls


def test_language_code_saved_as_importable_module(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(launcher, "download_from_github", lambda url, out: saved.append(out))
    monkeypatch.setattr(launcher, "load_env_variables", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BRANCH", "dev")
    monkeypatch.setattr(sys, "argv", ["launcher.py", "-u", "-x"])
    launcher.main()
    assert "subgen-dev.py" in saved
    assert "language_code.py" in saved
