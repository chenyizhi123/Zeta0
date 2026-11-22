#!/usr/bin/env python3
"""Copy the current source into an independent directory, without Git history."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


MAX_FILE_BYTES = 2_000_000


def _git(root: Path, *args: str, input: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], input=input,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    ).stdout


def source_files(root: Path) -> list[Path]:
    """Select current tracked/new files using both Git ignore and archive rules."""
    root = root.resolve(strict=True)
    candidates = sorted(set(filter(None, _git(
        root, "ls-files", "--cached", "--others", "--exclude-standard", "-z",
    ).decode().split("\0"))))
    if not candidates:
        raise ValueError("no source files found")
    encoded = ("\0".join(candidates) + "\0").encode()
    # --no-index also applies local exclusions to previously tracked files.
    ignored = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--no-index", "-z", "--stdin"],
        input=encoded, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if ignored.returncode not in (0, 1):
        raise RuntimeError("could not resolve source exclusions")
    excluded = set(ignored.stdout.decode().split("\0"))
    # Directory archive attributes suppress the whole subtree. check-attr does
    # not inherit them, so query ancestors explicitly as git archive would.
    attribute_paths = sorted({
        str(parent) for name in candidates
        for parent in (Path(name), *Path(name).parents) if str(parent) != "."
    })
    attrs = _git(
        root, "check-attr", "-z", "--stdin", "export-ignore",
        input=("\0".join(attribute_paths) + "\0").encode(),
    )
    fields = attrs.decode().split("\0")
    for name, _attribute, value in zip(fields[::3], fields[1::3], fields[2::3]):
        if value == "set":
            excluded.add(name)
    result = []
    for name in candidates:
        path = root / name
        if any(str(parent) in excluded for parent in (Path(name), *Path(name).parents)):
            continue
        if path.is_symlink():
            raise ValueError(f"source symlink must be reviewed: {name}")
        if path.is_dir():
            continue  # Gitlinks are external dependencies, not source files.
        if not path.exists():
            continue  # A working-tree deletion must not resurrect HEAD contents.
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"invalid source file: {name}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"oversized source file: {name}")
        result.append(path.relative_to(root))
    return result


def export_source(root: Path, output: Path) -> dict:
    root = root.resolve(strict=True)
    output = output.absolute()
    if output.is_symlink() or output.exists():
        raise FileExistsError("source output must be a new directory")
    output = output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("source output must be outside the repository")
    paths = source_files(root)
    if not paths:
        raise ValueError("source selection is empty")
    output.mkdir(parents=True, exist_ok=False)
    try:
        records = []
        for relative in paths:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, destination)
            contents = destination.read_bytes()
            records.append({
                "path": relative.as_posix(), "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            })
        manifest = {"format": "zeta0.source.v1", "files": records}
        (output / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(output)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = export_source(root, args.output)
    size = sum(entry["bytes"] for entry in manifest["files"])
    print(f"Exported {len(manifest['files'])} files ({size:,} bytes) to {args.output.resolve()}")


if __name__ == "__main__":
    main()
