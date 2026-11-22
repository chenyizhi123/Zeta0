"""Read-only source inventory and syntax/credential-pattern scan.

Local runs, datasets, environments and external checkouts are excluded.

Outputs paths, line numbers and categories, never matching secret values. This is
not an execution test, entropy scanner, license audit or publication clearance.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import stat
import subprocess
from collections import Counter
from pathlib import Path

EXCLUDED = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    "node_modules",
    "build",
    "dist",
    "checkpoints",
    "runs",
    "wandb",
}
ROOT_EXCLUDED = {
    "posttrain_runs",
    "posttrain_assets",
    "eval_runs",
    "deploy_packages",
    "diagnostics",
    "chenyz",
    "data",
    "models",
    "paper",
    "third_party",
}
TEXT_LIMIT = 2_000_000
CREDENTIAL_PATTERNS = {
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "github_token": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "hf_token": r"\bhf_[A-Za-z0-9]{30,}\b",
    "openai_key": r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b",
    "aws_access_key": r"\bAKIA[0-9A-Z]{16}\b",
}


def json_config_loads(text: str, *, comments: bool = False):
    """Validate JSON, optionally allowing JSONC comments and trailing commas.

    String tokens are retained verbatim. Comments become whitespace, not empty
    text, so they cannot join invalid adjacent numbers or keywords into a value.
    The stdlib decoder remains responsible for the actual JSON grammar.
    """
    if comments:
        strings_and_comments = r'"(?:[^"\\]|\\.)*"|//[^\r\n]*|/\*[\s\S]*?\*/'
        text = re.sub(
            strings_and_comments,
            lambda match: (
                match[0] if match[0].startswith('"')
                else re.sub(r'[^\r\n]', ' ', match[0])
            ),
            text,
        )
        tokens = list(re.finditer(r'"(?:[^"\\]|\\.)*"|[^\s]', text))
        chars = list(text)
        for index, token in enumerate(tokens):
            if (
                token[0] == ',' and 0 < index < len(tokens) - 1
                and tokens[index + 1][0] in ('}', ']')
                and tokens[index - 1][0] not in ('{', '[', ',', ':')
            ):
                chars[token.start()] = ' '
        text = ''.join(chars)

    def reject_constant(value):
        raise ValueError("Non-finite constants are not JSON")

    return json.loads(text, parse_constant=reject_constant)


def is_jsonc_config(path: Path) -> bool:
    return (
        path.suffix == '.jsonc'
        or path.name == 'pyrightconfig.json'
        or (path.parent.name == '.vscode' and path.name in {
            'settings.json', 'launch.json', 'tasks.json', 'extensions.json',
        })
    )


def scan(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("Scan root must be an existing directory")
    counts = Counter()
    areas = Counter()
    findings = []

    def report_directory_error(error: OSError) -> None:
        # os.walk otherwise silently omits inaccessible trees, including the root.
        path = Path(error.filename) if error.filename else root
        counts["directories_not_read"] += 1
        findings.append(dict(path=str(path.relative_to(root)), kind="directory_read_error"))

    for parent, dirs, names in os.walk(root, followlinks=False, onerror=report_directory_error):
        dirs[:] = sorted(
            d
            for d in dirs
            if d not in EXCLUDED
            and not (Path(parent) == root and d in ROOT_EXCLUDED)
            and not d.startswith((".venv", ".runs"))
            and not d.endswith(".egg-info")
            and not (Path(parent) / d).is_symlink()
        )
        for name in sorted(names):
            path = Path(parent) / name
            relative = str(path.relative_to(root))
            if path.is_symlink():
                counts["symlinks_not_read"] += 1
                continue
            counts["files"] += 1
            areas[path.relative_to(root).parts[0] if "/" in relative else "(root)"] += 1
            try:
                metadata = path.stat()
            except OSError:
                findings.append(dict(path=relative, kind="file_read_error"))
                continue
            if not stat.S_ISREG(metadata.st_mode):
                counts["non_regular_not_read"] += 1
                findings.append(dict(path=relative, kind="non_regular_file"))
                continue
            if metadata.st_size > TEXT_LIMIT:
                counts["oversize_not_read"] += 1
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError:
                counts["non_utf8_not_read"] += 1
                continue
            except OSError:
                findings.append(dict(path=relative, kind="file_read_error"))
                continue
            counts["text_files"] += 1
            if path.suffix == ".py":
                counts["python"] += 1
                try:
                    ast.parse(text, filename=relative)
                except SyntaxError as exc:
                    findings.append(
                        dict(path=relative, line=exc.lineno, kind="python_syntax")
                    )
            if path.suffix in {".json", ".jsonc"}:
                dialect = "jsonc" if is_jsonc_config(path) else "json"
                counts[dialect] += 1
                try:
                    json_config_loads(text, comments=dialect == "jsonc")
                except ValueError:
                    findings.append(dict(path=relative, kind=f"{dialect}_syntax"))
            if path.suffix in {".sh", ".sbatch"}:
                counts["shell"] += 1
                try:
                    result = subprocess.run(
                        ["bash", "-n", str(path)],
                        env={**os.environ, "BASH_ENV": "/dev/null", "ENV": "/dev/null"},
                        capture_output=True,
                        timeout=10,
                    )
                    if result.returncode:
                        findings.append(dict(path=relative, kind="shell_syntax"))
                except (OSError, subprocess.TimeoutExpired):
                    findings.append(dict(path=relative, kind="shell_check_unavailable"))
            for number, line in enumerate(text.splitlines(), 1):
                if re.search(r"/(?:inspire|hfm|home|mnt|opt)/", line):
                    findings.append(
                        dict(path=relative, line=number, kind="machine_path")
                    )
                for kind, pattern in CREDENTIAL_PATTERNS.items():
                    if re.search(pattern, line):
                        findings.append(dict(path=relative, line=number, kind=kind))
    return dict(counts=dict(counts), areas=dict(areas), findings=findings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    report = scan(args.root)
    print(json.dumps(report, indent=2))
    # Machine-local paths are review items, not proof of a defective executable.
    return int(any(item["kind"] != "machine_path" for item in report["findings"]))


if __name__ == "__main__":
    raise SystemExit(main())
