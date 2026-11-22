"""Exercise installed Click with disposable editor/CLI fixtures only."""

import json
import shlex
import sys

import pytest
from click.testing import CliRunner
from dotenv.cli import cli


@pytest.mark.parametrize("filename", [
    "ordinary file.txt", "literal$(printf substituted).txt", "literal$ZETA0_CLICK_TEST.txt",
])
def test_editor_receives_literal_filename(tmp_path, filename, run_isolated_process):
    source = tmp_path / filename
    source.write_text("unchanged fixture")
    output = tmp_path / "arguments.json"
    editor = tmp_path / "editor fixture.py"
    editor.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\n"
    )
    command = shlex.join([sys.executable, str(editor), str(output)])
    # No user EDITOR is consulted; the fake editor only records argv and exits.
    result = run_isolated_process([
        sys.executable, "-c",
        "import click,sys; click.edit(filename=sys.argv[1], editor=sys.argv[2], "
        "env={'ZETA0_CLICK_TEST': 'expanded'})",
        str(source), command,
    ], timeout=10, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == [str(source)]
    assert source.read_text() == "unchanged fixture"


@pytest.mark.parametrize("key,exit_code,output", [
    ("FIXTURE_VALUE", 0, "fixture\n"), ("MISSING", 1, ""),
])
def test_real_dotenv_cli_read_boundary(tmp_path, key, exit_code, output):
    path = tmp_path / ".env"
    path.write_text("FIXTURE_VALUE=fixture\n")
    result = CliRunner().invoke(cli, ["-f", str(path), "get", key])
    assert result.exit_code == exit_code
    assert result.output == output
    assert path.read_text() == "FIXTURE_VALUE=fixture\n"
