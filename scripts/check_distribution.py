"""Inspect the built wheel's boundaries without extracting or importing it."""

import argparse
import stat
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def check_wheel(path: Path) -> None:
    with ZipFile(path) as wheel:
        names = wheel.namelist()
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Expected exactly one distribution metadata file")
        if len(metadata[0].split('/')) != 2:
            raise ValueError("Distribution metadata must be at the archive root")
        info_root = metadata[0].split("/")[0]
        if not info_root.startswith('zeta0_humanoid-'):
            raise ValueError("Unexpected distribution metadata directory")
        info = BytesParser().parsebytes(wheel.read(metadata[0]))
        if info["Name"] != "zeta0-humanoid":
            raise ValueError("Unexpected distribution name")
        allowed = {"psi", "zeta0", info_root}
        seen = set()
        files = set()
        for member in wheel.infolist():
            name = member.filename
            parts = PurePosixPath(name).parts
            canonical = str(PurePosixPath(name))
            kind = stat.S_IFMT(member.external_attr >> 16)
            if (
                not parts
                or parts[0] not in allowed
                or ".." in parts
                or name.startswith("/")
                or name != canonical + ('/' if member.is_dir() else '')
                or '\\' in name or ':' in name
                or any(ord(char) < 32 for char in name)
                or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
                or (kind == stat.S_IFDIR and not member.is_dir())
                or (kind == stat.S_IFREG and member.is_dir())
            ):
                raise ValueError(f"Unexpected wheel member: {name!r}")
            if canonical in seen:
                raise ValueError(f"Duplicate wheel member: {name!r}")
            seen.add(canonical)
            if not member.is_dir():
                files.add(canonical)
        for name in seen:
            if any(str(parent) in files for parent in PurePosixPath(name).parents):
                raise ValueError(f"Conflicting file/directory wheel member: {name!r}")
        for required in (
            "psi/__init__.py",
            "psi/data/predictor_pretrain/runtime_contract.py",
            "zeta0/__init__.py",
        ):
            if required not in names:
                raise ValueError(f"Missing core module: {required}")
        for notice in ("LICENSE", "NOTICE", "Apache-2.0.txt", "THIRD_PARTY_NOTICES.md"):
            if not any(
                name.startswith(info_root + "/") and name.endswith("/" + notice)
                for name in names
            ):
                raise ValueError(f"Missing attribution file: {notice}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    check_wheel(args.wheel)
    print(f"{args.wheel.name}: distribution boundaries and notices PASS")


if __name__ == "__main__":
    main()
