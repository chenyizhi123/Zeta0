"""Read-only verification of optional source checkouts; no fetch or installation."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def check_source(root: Path, entry: dict) -> list[str]:
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return ["source path must be a nonempty relative path without traversal"]
    path = root / relative
    if any(
        (root / Path(*relative.parts[:i])).is_symlink()
        for i in range(1, len(relative.parts) + 1)
    ):
        return ["source path must not traverse a symlink"]
    if not re.fullmatch(r"[0-9a-f]{40}", entry["commit"]):
        return ["source commit must be a full lowercase SHA-1"]
    if not path.is_dir():
        return ["source directory is missing"]

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode:
            # Do not echo remote URLs, credentials or arbitrary Git diagnostics.
            raise ValueError(f"git {args[0]} failed (exit {result.returncode})")
        return result.stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != path.resolve():
            return ["directory is not its own Git checkout"]
        errors = []
        if git("rev-parse", "HEAD") != entry["commit"]:
            errors.append("HEAD does not match the recorded commit")
        if git("remote", "get-url", "origin") != entry["url"]:
            errors.append("origin does not match the recorded URL")
        if git("status", "--porcelain", "--untracked-files=all"):
            errors.append("source checkout has local changes or untracked files")
        return errors
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        return [str(error) if isinstance(error, ValueError) else type(error).__name__]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names", nargs="*", help="source names; defaults to all recorded sources"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with (root / "third_party/sources.toml").open("rb") as handle:
        sources = tomllib.load(handle)
    names = args.names or list(sources)
    unknown = set(names) - sources.keys()
    if unknown:
        parser.error("unknown source names: " + ", ".join(sorted(unknown)))
    failed = False
    for name in names:
        errors = check_source(root, sources[name])
        failed |= bool(errors)
        print(f"{name}: " + ("; ".join(errors) if errors else "pinned clean checkout"))
    print(
        "Source-only check: does not validate LFS assets, licenses, installation or parent gitlinks."
    )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
