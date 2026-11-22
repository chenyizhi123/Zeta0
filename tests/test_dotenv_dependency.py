"""Real temporary-file checks; never load or modify the project's .env."""

import os

import pytest
from dotenv import dotenv_values, load_dotenv, set_key, unset_key


@pytest.mark.parametrize("operation", ["set", "unset"])
def test_rewrite_replaces_symlink_without_modifying_target(tmp_path, operation):
    target = tmp_path / "unrelated-config"
    original = "KEEP=original\nREMOVE=present\n"
    target.write_text(original)
    link = tmp_path / ".env"
    link.symlink_to(target)
    if operation == "set":
        assert set_key(link, "ADDED", "new value") == (True, "ADDED", "new value")
    else:
        assert unset_key(link, "REMOVE") == (True, "REMOVE")
    assert target.read_text() == original
    assert not link.is_symlink()
    values = dotenv_values(link)
    assert values["KEEP"] == "original"
    if operation == "set":
        assert values["ADDED"] == "new value"
    else:
        assert "REMOVE" not in values


@pytest.mark.parametrize("override", [False, True])
def test_load_preserves_override_and_interpolation(tmp_path, monkeypatch, override):
    key = "ZETA0_DOTENV_TEST_VALUE"
    derived = "ZETA0_DOTENV_TEST_DERIVED"
    monkeypatch.setenv(key, "existing")
    monkeypatch.delenv(derived, raising=False)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    path = tmp_path / ".env"
    path.write_text(f"{key}=file\n{derived}=${{{key}}}/suffix\n")
    assert load_dotenv(path, override=override)
    expected = "file" if override else "existing"
    assert os.environ[key] == expected
    assert os.environ[derived] == expected + "/suffix"
