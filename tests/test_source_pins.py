"""Exercise source identity checks with real temporary Git repositories."""

import subprocess

import pytest

from scripts.check_source_pins import check_source


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "vendor"
    path.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    git("init", "-q")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "--allow-empty",
        "-m",
        "fixture",
    )
    git("remote", "add", "origin", "https://example.invalid/source.git")
    return (
        tmp_path,
        path,
        dict(
            path="vendor",
            url="https://example.invalid/source.git",
            commit=git("rev-parse", "HEAD"),
        ),
    )


def test_clean_checkout_matches(source):
    root, _, entry = source
    assert check_source(root, entry) == []


def test_mismatched_pin_and_origin_fail(source):
    root, _, entry = source
    errors = check_source(
        root, {**entry, "commit": "0" * 40, "url": "https://wrong.invalid"}
    )
    assert len(errors) == 2


def test_untracked_changes_fail(source):
    root, path, entry = source
    (path / "local.txt").write_text("user work")
    assert "local changes" in check_source(root, entry)[0]


def test_empty_child_cannot_borrow_parent_git(source):
    root, path, entry = source
    (path / "empty").mkdir()
    assert (
        "own Git checkout" in check_source(root, {**entry, "path": "vendor/empty"})[0]
    )


@pytest.mark.parametrize("path", ["../escape", "/outside", "."])
def test_invalid_paths_fail(source, path):
    root, _, entry = source
    assert check_source(root, {**entry, "path": path})


def test_symlink_checkout_rejected(source):
    root, path, entry = source
    (root / "alias").symlink_to(path, target_is_directory=True)
    assert "symlink" in check_source(root, {**entry, "path": "alias"})[0]
