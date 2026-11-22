"""Export real Git working-tree changes without history or local artifacts."""

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.export_source import export_source, source_files


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("*.pt\n.env\nsource_manifest.json\n")
    (root / ".gitattributes").write_text("/private export-ignore\n")
    (root / "core.py").write_text("old contents\n")
    (root / "removed.py").write_text("retired implementation\n")
    (root / "tracked.pt").write_bytes(b"old tracked weights")
    subprocess.run(["git", "-C", str(root), "add", "-f", "."], check=True)
    (root / "core.py").write_text("current contents\n")
    (root / "removed.py").unlink()
    (root / "new.py").write_text("new mainline module\n")
    (root / ".env").write_text("LOCAL=private\n")
    private = root / "private"
    private.mkdir()
    (private / "notes.txt").write_text("local notes\n")
    return root


def test_export_uses_working_tree_and_both_exclusion_rules(repository, tmp_path):
    output = tmp_path / "source"
    manifest = export_source(repository, output)
    names = {record["path"] for record in manifest["files"]}
    assert {"core.py", "new.py", ".gitignore", ".gitattributes"} == names
    assert (output / "core.py").read_text() == "current contents\n"
    assert not (output / ".git").exists()
    assert json.loads((output / "source_manifest.json").read_text()) == manifest
    for record in manifest["files"]:
        assert record["sha256"] == hashlib.sha256((output / record["path"]).read_bytes()).hexdigest()


def test_export_refuses_source_symlinks_before_writing(repository, tmp_path):
    (repository / "link.py").symlink_to(repository / "core.py")
    with pytest.raises(ValueError, match="symlink"):
        export_source(repository, tmp_path / "source")
    assert not (tmp_path / "source").exists()


def test_export_does_not_overwrite_or_nest_inside_repository(repository, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError):
        export_source(repository, existing)
    assert sentinel.read_text() == "unchanged"
    with pytest.raises(ValueError, match="outside"):
        export_source(repository, repository / "nested-source")


def test_export_refuses_oversized_candidates(repository):
    (repository / "large.bin").write_bytes(b"x" * 2_000_001)
    with pytest.raises(ValueError, match="oversized"):
        source_files(repository)


def test_local_artifact_symlinks_stay_out_of_git_and_export(repository, tmp_path):
    maintained_root = Path(__file__).resolve().parents[1]
    (repository / ".gitignore").write_bytes((maintained_root / ".gitignore").read_bytes())
    artifacts = tmp_path / "local-artifacts"
    artifacts.mkdir()
    (artifacts / "private.txt").write_text("local experiment contents")
    names = {
        "data", "models", "posttrain_assets", "posttrain_runs", "eval_runs",
        "runs", "checkpoints", "deploy_packages", "diagnostics", "chenyz", "paper",
    }
    for name in names:
        (repository / name).symlink_to(artifacts, target_is_directory=True)
    for package in ("data", "models"):
        source = repository / "src" / "psi" / package / "__init__.py"
        source.parent.mkdir(parents=True)
        source.write_text("# maintained source\n")
    subprocess.run(["git", "-C", str(repository), "add", "--all"], check=True)
    tracked = set(subprocess.check_output(
        ["git", "-C", str(repository), "ls-files", "-z"],
    ).decode().strip("\0").split("\0"))
    assert not names.intersection(tracked)
    manifest = export_source(repository, tmp_path / "source")
    exported = {record["path"] for record in manifest["files"]}
    assert {"src/psi/data/__init__.py", "src/psi/models/__init__.py"} <= exported
    assert not names.intersection(Path(name).parts[0] for name in exported)
