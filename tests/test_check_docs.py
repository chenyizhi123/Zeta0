import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check_docs.py"
check_document = runpy.run_path(str(CHECKER))["check_document"]


def test_local_relative_root_and_encoded_links(tmp_path):
    target = tmp_path / "Some Guide.md"
    target.write_text("# Setup\n## More `detail`\n## Setup\n", encoding="utf-8")
    source = tmp_path / "nested/README.md"
    source.parent.mkdir()
    source.write_text(
        "[one](../Some%20Guide.md#setup)\n"
        "[two](</Some Guide.md#more-detail>)\n"
        "[duplicate](../Some%20Guide.md#setup-1)\n"
        "# Here\n[self](#here)\n",
        encoding="utf-8",
    )
    assert check_document(source, tmp_path) == []


@pytest.mark.parametrize(
    "link,reason",
    [
        ("missing.md", "missing target"),
        ("target.md#missing", "missing heading anchor"),
        ("../outside.md", "target escapes repository"),
    ],
)
def test_broken_links_report_source_line(tmp_path, link, reason):
    source = tmp_path / "README.md"
    source.write_text(f"# Title\n[broken]({link})\n", encoding="utf-8")
    (tmp_path / "target.md").write_text("# Exists\n", encoding="utf-8")
    assert check_document(source, tmp_path) == [f"README.md:2: {reason}: {link}"]


def test_external_links_and_fenced_examples_are_not_checked(tmp_path):
    source = tmp_path / "README.md"
    source.write_text(
        "[web](https://example.invalid/no-network)\n"
        "[mail](mailto:someone@example.invalid)\n"
        "````markdown\n```\n[example](missing.md)\n````\n"
        "~~~markdown\n[example](also-missing.md)\n~~~\n",
        encoding="utf-8",
    )
    assert check_document(source, tmp_path) == []


def test_symlink_cannot_escape_repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "outside.md").write_text("# Outside\n", encoding="utf-8")
    (root / "linked.md").symlink_to(tmp_path / "outside.md")
    source = root / "README.md"
    source.write_text("[bad](linked.md)\n", encoding="utf-8")
    assert "target escapes repository" in check_document(source, root)[0]


def test_cli_failure_and_success_from_other_directory(tmp_path):
    # Run the real entry point in a tiny repository; no parser or subprocess mocks.
    script = tmp_path / "scripts/check_docs.py"
    script.parent.mkdir()
    script.write_text(CHECKER.read_text(encoding="utf-8"), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "missing maintained document" in result.stdout
    for name in (
        "README.md",
        "docs/README.md",
        "docs/reproducibility.md",
        "AGENTS.md",
        "CONTRIBUTING.md",
        "THIRD_PARTY_NOTICES.md",
        "SECURITY.md",
        "configs/predictive_world/README.md",
        "real/README.md",
        "configs/psi0/README.md",
        "configs/gr00t_n1d7/README.md",
        ".github/pull_request_template.md",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Guide\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_local_dependency_documentation_is_outside_owned_link_checks(tmp_path):
    from scripts.check_docs import maintained_document
    assert maintained_document(tmp_path / 'real/g1/README.md', tmp_path)
    assert not maintained_document(tmp_path / 'real/g1/GR00T-WholeBodyControl/gear_sonic/data/README.md', tmp_path)
    assert not maintained_document(tmp_path / 'real/g1/external_dependencies/sdk/README.md', tmp_path)
