#!/usr/bin/env python3
"""Check or apply the source-bound SONIC ACK patch; never build or start a robot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path, *, apply: bool = False) -> str:
    root = root.resolve(strict=True)
    patch_dir = Path(__file__).resolve().parents[2] / "real/patches"
    patch = patch_dir / "sonic_velocity_fallback_ack_v1.patch"
    provenance = json.loads(patch.with_suffix(".provenance.json").read_text())
    if digest(patch) != provenance["patch_sha256"]:
        raise ValueError("SONIC source patch differs from its provenance")
    states = []
    for item in provenance["files"]:
        path = (root / item["path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("SONIC patch path escapes the selected checkout")
        actual = digest(path)
        if actual == item["source_sha256"]:
            states.append("source")
        elif actual == item["patched_sha256"]:
            states.append("patched")
        else:
            raise ValueError(f"SONIC source differs from reviewed ZIP: {item['path']}")
    if set(states) == {"patched"}:
        return "ACK v1 source patch is already applied; no build or controller was started."
    if set(states) != {"source"}:
        raise ValueError("SONIC checkout contains a partial ACK patch; reconcile it before applying")
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True)
    if not apply:
        return "Source hashes and patch applicability passed; no files changed."
    subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
    for item in provenance["files"]:
        if digest(root / item["path"]) != item["patched_sha256"]:
            raise ValueError(f"Patched SONIC source hash differs: {item['path']}")
    return "ACK v1 source patch applied and hashes verified; no build or controller was started."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply after all source hashes pass; default only checks")
    args = parser.parse_args()
    try:
        print(prepare(args.sonic_root, apply=args.apply))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
