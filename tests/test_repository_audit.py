import json
import os
import runpy
import shlex
from pathlib import Path

import pytest

scan = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/audit_repository.py")
)["scan"]


def test_local_artifact_names_do_not_hide_nested_model_and_data_source(tmp_path):
    for name in ("data", "models"):
        artifact = tmp_path / name
        artifact.mkdir()
        (artifact / "local.py").write_text("def !\n")
        source = tmp_path / "src/psi" / name
        source.mkdir(parents=True)
        (source / "core.py").write_text("def !\n")
    report = scan(tmp_path)
    assert {item["path"] for item in report["findings"]} == {
        "src/psi/data/core.py", "src/psi/models/core.py",
    }


def test_includes_vendor_source_but_does_not_execute_it(tmp_path):
    vendor = tmp_path / "src/vendor"
    vendor.mkdir(parents=True)
    (vendor / "model.py").write_text("raise RuntimeError('must not execute')\n")
    script = tmp_path / "job.sh"
    script.write_text("touch " + shlex.quote(str(tmp_path / "should-not-exist")) + "\n")
    for directory in (".venv-check", "build", "__pycache__"):
        ignored = tmp_path / directory
        ignored.mkdir()
        (ignored / "invalid.py").write_text("def !\n")
    result = scan(tmp_path)
    assert result["counts"]["python"] == 1
    assert result["counts"]["shell"] == 1
    assert result["findings"] == []
    assert not (tmp_path / "should-not-exist").exists()


def test_reports_categories_without_secret_values(tmp_path):
    secret = "ghp_" + "A" * 36
    (tmp_path / "config.txt").write_text(secret + "\n")
    (tmp_path / "bad.py").write_text("def !\n")
    (tmp_path / "bad.json").write_text("{")
    result = scan(tmp_path)
    kinds = {item["kind"] for item in result["findings"]}
    assert kinds == {"github_token", "python_syntax", "json_syntax"}
    assert secret not in json.dumps(result)


def test_does_not_follow_external_symlinks(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def !\n")
    (root / "linked.py").symlink_to(outside)
    result = scan(root)
    assert result["counts"]["symlinks_not_read"] == 1
    assert result["findings"] == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_fifo_is_reported_without_blocking(tmp_path, run_isolated_process):
    os.mkfifo(tmp_path / "pipe.py")
    scanner = Path(__file__).resolve().parents[1] / "scripts/audit_repository.py"
    import sys

    result = run_isolated_process(
        [sys.executable, str(scanner), "--root", str(tmp_path)], timeout=5
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["counts"]["non_regular_not_read"] == 1
    assert report["findings"] == [{"path": "pipe.py", "kind": "non_regular_file"}]


def test_unreadable_file_is_reported_and_scan_continues(tmp_path, monkeypatch):
    unreadable = tmp_path / "a.txt"
    unreadable.write_text("contents")
    (tmp_path / "z.py").write_text("def !\n")
    original_read = Path.read_text

    def fail_one_file(path, *args, **kwargs):
        if path == unreadable:
            raise PermissionError("private diagnostic must not be printed")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_one_file)
    report = scan(tmp_path)
    assert {item["kind"] for item in report["findings"]} == {
        "file_read_error", "python_syntax"
    }
    assert "private diagnostic" not in json.dumps(report)


@pytest.mark.parametrize("location", [".", "unreadable"])
def test_directory_traversal_failure_is_not_a_clean_scan(tmp_path, monkeypatch, location):
    denied = tmp_path / location
    denied.mkdir(exist_ok=True)
    (tmp_path / "visible.py").write_text("def !\n")
    original_scandir = os.scandir

    def fail_directory(path):
        if Path(path) == denied:
            raise PermissionError(13, "private diagnostic must not be printed", str(path))
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", fail_directory)
    report = scan(tmp_path)
    assert {"path": location, "kind": "directory_read_error"} in report["findings"]
    assert report["counts"]["directories_not_read"] == 1
    if location != ".":
        assert report["counts"]["python"] == 1
        assert any(item["kind"] == "python_syntax" for item in report["findings"])
    assert "private diagnostic" not in json.dumps(report)


def test_cli_fails_when_root_cannot_be_traversed(tmp_path, monkeypatch, capsys):
    import sys

    scanner = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/audit_repository.py")
    )
    original_scandir = os.scandir

    def fail_directory(path):
        if Path(path) == tmp_path:
            raise PermissionError(13, "unavailable", str(path))
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", fail_directory)
    monkeypatch.setattr(sys, "argv", ["audit_repository.py", "--root", str(tmp_path)])
    assert scanner["main"]() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["findings"] == [{"path": ".", "kind": "directory_read_error"}]
