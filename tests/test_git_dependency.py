"""Local-only GitPython security and W&B provenance compatibility checks."""

from pathlib import Path

from git import Repo
from git.objects.submodule.base import Submodule
from wandb.sdk.lib.gitlib import GitRepo


def test_submodule_config_cannot_include_unrelated_local_file(tmp_path):
    included = tmp_path / "unrelated.ini"
    included.write_text('[submodule "unit"]\nmarker = harmless test contents\n')
    modules = tmp_path / ".gitmodules"
    modules.write_text(
        f'[include]\npath = {included}\n[submodule "unit"]\n'
        'path = unit\nurl = https://example.invalid/unit.git\n'
    )

    with Repo.init(tmp_path) as repo:
        submodule = Submodule(repo, b"\0" * 20, path="unit", name="unit")
        with submodule.config_reader() as config:
            assert config.get_value("path") == "unit"
            assert not config.has_option("marker")


def test_wandb_local_source_metadata_survives_gitpython_upgrade(tmp_path):
    # Only this disposable repository is initialized/committed; no network or W&B run.
    with Repo.init(tmp_path) as repo:
        with repo.config_writer() as config:
            config.set_value("user", "name", "Fixture")
            config.set_value("user", "email", "fixture@example.invalid")
            config.set_value("core", "hooksPath", "/dev/null")
        source = tmp_path / "source.txt"
        source.write_text("original\n")
        repo.index.add(["source.txt"])
        commit = repo.index.commit("fixture")
        repo.create_remote("origin", "https://example.invalid/oracle-fixture.git")

        recorded = GitRepo(root=str(tmp_path), lazy=False)
        try:
            assert recorded.enabled
            # W&B 0.31 replaced its GitPython-backed root/repo accessors with
            # root_dir and subprocess Git. Check the same provenance for both
            # the locked SDK and newer versions allowed by the dependency floor.
            root = recorded.root_dir if hasattr(recorded, "root_dir") else recorded.root
            assert Path(root).resolve() == tmp_path.resolve()
            assert recorded.last_commit == commit.hexsha
            assert recorded.remote_url == "https://example.invalid/oracle-fixture.git"
            assert not recorded.dirty
            source.write_text("changed\n")
            assert recorded.dirty
            (tmp_path / "untracked.txt").write_text("untracked\n")
            assert recorded.is_untracked("untracked.txt")
        finally:
            gitpython_repo = getattr(recorded, "repo", None)
            if gitpython_repo is not None:
                gitpython_repo.close()
